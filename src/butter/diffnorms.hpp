namespace butter{

__global__ void sumreduce(float* dst, float* src, int width){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is 3*sizeof(float)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
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

__global__ void sumreducenorm(float* dst, float* src, int width){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is sizeof(float)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    __shared__ float sharedmem[1024*3];

    if (x >= width){
        sharedmem[thx] = 0.0f;
        sharedmem[1024+thx] = 0.0f;
        sharedmem[1024*2+thx] = 0.0f;
    } else {
        sharedmem[thx] = powf(abs(src[x]), 2);
        sharedmem[1024+thx] = powf(abs(src[x]), 3);
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

std::tuple<float, float, float> diffmapscore(float* diffmap, float* temp, float* temp2, int width, hipEvent_t event_d, hipStream_t stream){
    bool first = true;
    int basewidth = width;
    float* src = diffmap;
    float* temps[2] = {temp, temp2};
    int oscillate = 0;
    int th_x, bl_x;
    while (width > 1024){
        th_x = 1024;
        bl_x = (width - 1)/th_x + 1;
        if (first){
            sumreducenorm<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width);
        } else {
            sumreduce<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width);
        }
        src = temps[oscillate];
        oscillate ^= 1;
        first = false;
        width = bl_x;
    }
    float* back_to_cpu = (float*)malloc(sizeof(float)*width*3);
    if (!back_to_cpu) throw std::bad_alloc();
    float resnorm2 = 0.0f;
    float resnorm3 = 0.0f;
    float resnorminf = 0.0f;
    hipMemcpyDtoHAsync(back_to_cpu, src, sizeof(float)*width*3, stream);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    for (int i = 0; i < width; i++){
        if (first){
            resnorm2 += std::pow(std::abs(back_to_cpu[i]), 2);
            resnorm3 += std::pow(std::abs(back_to_cpu[i]), 3);
            resnorminf = std::max(resnorminf, std::abs(back_to_cpu[i]));
        } else {
            resnorm2 += std::abs(back_to_cpu[i]);
            resnorm3 += std::abs(back_to_cpu[i+width]);
            resnorminf = std::max(resnorminf, std::abs(back_to_cpu[i+2*width]));
        }
    }

    free(back_to_cpu);
    resnorm2 = std::powf(resnorm2/basewidth, 1.0/2.0);
    resnorm3 = std::pow(resnorm3/basewidth, 1.0/3.0);
    return std::make_tuple(resnorm2, resnorm3, resnorminf);
}

}