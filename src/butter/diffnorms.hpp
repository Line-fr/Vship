namespace butter{

int64_t allocsizeScore(int64_t width, int64_t height){
    int64_t w = width*height;
    while (w > 1024){
        w = (w - 1)/1024 + 1;
    }
    return w*3;
}

__global__ void sumreduce(float* dst, float* src, int64_t width){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is 3*sizeof(float)*threadnum at least
    const int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    const int64_t thx = threadIdx.x;
    const int64_t threadnum = blockDim.x;
    
    __shared__ float sharedmem[1024*3];

    if (x >= width){
        sharedmem[thx] = 0.0f;
        sharedmem[1024+thx] = 0.0f;
        sharedmem[1024*2+thx] = 0.0f;
    } else {
        sharedmem[thx] = src[x];
        sharedmem[1024+thx] = src[x+width];
        sharedmem[1024*2+thx] = src[x+2*width];
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] += sharedmem[thx+next];
            sharedmem[thx + 1024] += sharedmem[thx+next + 1024];
            sharedmem[thx + 2*1024] = max(sharedmem[thx + 2*1024], sharedmem[thx+next + 2*1024]);
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0];
        dst[blockIdx.x + gridDim.x] = sharedmem[1024];
        dst[blockIdx.x + 2*gridDim.x] = sharedmem[2*1024];
    }
}

__global__ void sumreducenorm(float* dst, float* src, int64_t width, int Qnorm){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is sizeof(float)*threadnum at least
    const int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    const int64_t thx = threadIdx.x;
    const int64_t threadnum = blockDim.x;
    
    __shared__ float sharedmem[1024*3];

    if (x >= width){
        sharedmem[thx] = 0.0f;
        sharedmem[1024+thx] = 0.0f;
        sharedmem[1024*2+thx] = 0.0f;
    } else {
        //instead of norm2 we will have Qnorm now!
        sharedmem[thx] = powf(src[x], Qnorm);
        sharedmem[1024+thx] = abs(src[x])*src[x]*src[x];
        sharedmem[1024*2+thx] = abs(src[x]);
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] += sharedmem[thx+next];
            sharedmem[thx + 1024] += sharedmem[thx+next + 1024];
            sharedmem[thx + 2*1024] = max(sharedmem[thx + 2*1024], sharedmem[thx+next + 2*1024]);
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0];
        dst[blockIdx.x + gridDim.x] = sharedmem[1024];
        dst[blockIdx.x + 2*gridDim.x] = sharedmem[2*1024];
    }
}

std::tuple<double, double, double> diffmapscore(float* diffmap, float* temp, float* temp2, float* pinned, int64_t width, int Qnorm, hipStream_t stream){
    bool first = true;
    int64_t basewidth = width;
    float* src = diffmap;
    float* temps[2] = {temp, temp2};
    int oscillate = 0;
    int64_t th_x, bl_x;
    while (width > 1024){
        th_x = 1024;
        bl_x = (width - 1)/th_x + 1;
        if (first){
            sumreducenorm<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width, Qnorm);
        } else {
            sumreduce<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width);
        }
        src = temps[oscillate];
        oscillate ^= 1;
        first = false;
        width = bl_x;
    }
    float* back_to_cpu = pinned;
    double resnormQ = 0.0f;
    double resnorm3 = 0.0f;
    double resnorminf = 0.0f;
    hipMemcpyDtoHAsync(back_to_cpu, src, sizeof(float)*width*3, stream);

    hipStreamSynchronize(stream);

    for (int i = 0; i < width; i++){
        if (first){
            resnormQ += std::pow(std::abs(back_to_cpu[i]), Qnorm);
            resnorm3 += std::pow(std::abs(back_to_cpu[i]), 3);
            resnorminf = std::max(resnorminf, (double)std::abs(back_to_cpu[i]));
        } else {
            resnormQ += std::abs(back_to_cpu[i]);
            resnorm3 += std::abs(back_to_cpu[i+width]);
            resnorminf = std::max(resnorminf, (double)std::abs(back_to_cpu[i+2*width]));
        }
    }

    resnormQ = std::pow(resnormQ/basewidth, 1./(double)Qnorm);
    resnorm3 = std::pow(resnorm3/basewidth, 1./3.);
    return std::make_tuple(resnormQ, resnorm3, resnorminf);
}

}