namespace butter{

__global__ void loadGaussianKernel(float* gaussiankernel, float* gaussiankernel_integral, int gaussiansize, double sigma){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x] = expf(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
    if (x == 0){
        float acc = 0.;
        for (int i = 0; i <= 2*gaussiansize+1; i++){
            gaussiankernel_integral[i] = acc;
            acc += expf(-(gaussiansize-i)*(gaussiansize-i)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
        }
    }
}

//alows to precompute and reuse easily the same kernel array in gpu
class GaussianHandle{
    float* gaussiankernel_d;
    int* indices;
public:
    void init(){
        hipError_t erralloc = hipMalloc(&gaussiankernel_d, sizeof(float)*1024);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        indices = (int*)malloc(sizeof(int)*6);
        if (indices == NULL){
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }
        indices[0] = 0;
        const double sigmas[5] = {1.2, 1.56416327805, 2.7, 3.22489901262, 7.15593339443};

        for (int i = 0; i < 5; i++){
            const double sigma = sigmas[i];
            const int windowsize = std::max((int)(sigma*3), (int)8); //we use 8 to trigger the special gaussian blur when possible
            indices[i+1] = indices[i]+4*windowsize+3;
            const int th_x = std::min(windowsize*2+1, 256);
            const int bl_x = (windowsize*2)/th_x+1;
            loadGaussianKernel<<<dim3(bl_x), dim3(th_x), 0, 0>>>(get(i), getIntegral(i), windowsize, sigma);
            GPU_CHECK(hipGetLastError());
        }
        GPU_CHECK(hipDeviceSynchronize());
    }
    float* get(int i){
        return gaussiankernel_d+indices[i];
    }
    float* getIntegral(int i){
        return gaussiankernel_d+indices[i]+2*getWindow(i)+1;
    }
    int getWindow(int i){
        return ((indices[i+1]-indices[i])-3)/4;
    }
    void destroy(){
        GPU_CHECK(hipFree(gaussiankernel_d));
        free(indices);
    }
};

__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, float* gaussiankernel_integral, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;

    const int begin = max(x-gaussiansize, current_line*w);
    const int end = min(x+gaussiansize, (current_line+1)*w-1);
    float weight = gaussiankernel_integral[gaussiansize+end+1-x]-gaussiankernel_integral[gaussiansize+begin-x];

    float out = 0.0f;
    for (int i = begin; i <= end; i++){
        out += src[i]*gaussiankernel[gaussiansize+i-x];
    }
    dst[x] = out/weight;
}

__launch_bounds__(256)
__global__ void verticalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, float* gaussiankernel_integral, int gaussiansize){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y*blockDim.y;
    if (x >= w) return;
    if (y >= h) return;

    const int begin = max(y-gaussiansize, (int64_t)0);
    const int end = min(y+gaussiansize, (int64_t)h-1);
    float weight = gaussiankernel_integral[gaussiansize+end+1-y]-gaussiankernel_integral[gaussiansize+begin-y];;
    float out = 0.0f;
    for (int64_t i = begin; i <= end; i++){
        out += src[i * w + x]*gaussiankernel[gaussiansize+i-y];
    }
    dst[y*w+x] = out/weight;
}

__launch_bounds__(256)
__global__ void TiledGaussianBlur_Kernel(float* src, float* dst, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    int64_t originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly

    const int64_t blockwidth = (width-1)/32+1;

    const int64_t x = threadIdx.x + 32*(originalBl_X%blockwidth);
    const int64_t y = threadIdx.y + 32*(originalBl_X/blockwidth);
    const int64_t thx = threadIdx.x;
    const int64_t thy = threadIdx.y;

    __shared__ float tampon[48*48]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    #pragma unroll
    for (int region_x = 0; region_x < 48; region_x += 16){
        #pragma unroll
        for (int region_y = 0; region_y < 48; region_y += 16){
            tampon[(thy+region_y)*48+thx+region_x] = (tampon_base_x + thx +region_x >= 0 && tampon_base_x + thx +region_x < width && tampon_base_y + thy + region_y >= 0 && tampon_base_y + thy + region_y < height) ? src[(tampon_base_y+thy+region_y)*width + tampon_base_x+thx+region_x] : 0.f;
        }
    }
    __syncthreads();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread
    float tot[2];
    float out[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){

        //border handling precompute
        const int beg = max((int64_t)0, x-8+region_x*16)-(x-8+region_x*16);
        const int end2 = min(width, x+9+region_x*16)-(x-8+region_x*16);
        tot[region_x] = gaussiankernel_integral[end2]-gaussiankernel_integral[beg];

        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_x][region_y] += tampon[(thy+region_y*16)*48 + thx+i+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            tampon[(thy+region_y*16)*48 + thx+8+region_x*16] = out[region_x][region_y]/tot[region_x];
        }
    }
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 40][8 - 40] -> 4 pass per thread
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){
        
        //border handling precompute
        const int beg = max((int64_t)0, y-8+region_y*16)-(y-8+region_y*16);
        const int end2 = min(height, y+9+region_y*16)-(y-8+region_y*16);
        tot[region_y] = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            out[region_y][region_x] = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*16)*48 + thx+8+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            tampon[(thy+8+region_y*16)*48 + thx+8+region_x*16] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 2; region_y++){
            if (tampon_base_x + thx +8+region_x*16 >= 0 && tampon_base_x + thx +8+region_x*16 < width && tampon_base_y + thy +8+region_y*16 >= 0 && tampon_base_y + thy +8+region_y*16 < height) dst[(tampon_base_y+thy+8+region_y*16)*width + tampon_base_x+thx+8+region_x*16] = tampon[(thy+8+region_y*16)*48+thx+8+region_x*16];
        }
    }
}

void blur(float* mem_d, float* temp, int64_t width, int64_t height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);
    float* gaussianKernel_integral = gaussianHandle.getIntegral(i);

    int64_t wh = width*height;
    int64_t th_x = std::min((int64_t)256, wh);
    int64_t bl_x = (wh-1)/th_x + 1;

    int64_t verticalth_x = 32;
    int64_t verticalth_y = 8;
    int64_t verticalbl_x = (width-1)/verticalth_x+1;
    int64_t verticalbl_y = (height-1)/verticalth_y+1;

    horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, mem_d, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
    GPU_CHECK(hipGetLastError());
    verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(mem_d, temp, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
    GPU_CHECK(hipGetLastError());
}

void blur(float* dst, float* mem_d, float* temp, int64_t width, int64_t height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);
    float* gaussianKernel_integral = gaussianHandle.getIntegral(i);

    int64_t wh = width*height;

    if (gaussiansize == 8){ //special gaussian blur! It doesnt even use temp
        int64_t th_x = 16;
        int64_t th_y = 16;
        int64_t bl_x = (width-1)/(2*th_x)+1;
        int64_t bl_y = (height-1)/(2*th_y)+1;
        TiledGaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst, width, height, gaussianKernel, gaussianKernel_integral);
        GPU_CHECK(hipGetLastError());
    } else {
        int64_t th_x = std::min((int64_t)256, wh);
        int64_t bl_x = (wh-1)/th_x + 1;

        int64_t verticalth_x = 32;
        int64_t verticalth_y = 8;
        int64_t verticalbl_x = (width-1)/verticalth_x+1;
        int64_t verticalbl_y = (height-1)/verticalth_y+1;
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, mem_d, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
        GPU_CHECK(hipGetLastError());
        verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(dst, temp, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
        GPU_CHECK(hipGetLastError());
    }
}
void blurDstNoTemp(float* dst, float* mem_d, int64_t width, int64_t height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);
    float* gaussianKernel_integral = gaussianHandle.getIntegral(i);

    assert(gaussiansize == 8);
    //special gaussian blur! It doesnt even use temp
    int64_t th_x = 16;
    int64_t th_y = 16;
    int64_t bl_x = (width-1)/(2*th_x)+1;
    int64_t bl_y = (height-1)/(2*th_y)+1;
    TiledGaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst, width, height, gaussianKernel, gaussianKernel_integral);
    GPU_CHECK(hipGetLastError());
}

//all gaussian sigmas:
/*
1.2
1.56416327805
2.7
3.22489901262
7.15593339443
*/

}