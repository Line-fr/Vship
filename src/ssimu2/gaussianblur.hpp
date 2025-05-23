__device__ void GaussianSmartSharedLoadProduct(float3* tampon, const float3* src1, const float3* src2, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    tampon[thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src1[(tampon_base_y+thy)*width + tampon_base_x+thx]*src2[(tampon_base_y+thy)*width + tampon_base_x+thx] : makeFloat3(0.f, 0.f, 0.f);
    tampon[(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx]*src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : makeFloat3(0.f, 0.f, 0.f);
    tampon[thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src1[(tampon_base_y+thy)*width + tampon_base_x+thx+16]*src2[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : makeFloat3(0.f, 0.f, 0.f);
    tampon[(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16]*src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : makeFloat3(0.f, 0.f, 0.f);
    __syncthreads();
}

__device__ void GaussianSmartSharedLoad(float3* tampon, const float3* src, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    tampon[thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : makeFloat3(0.f, 0.f, 0.f);
    tampon[(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : makeFloat3(0.f, 0.f, 0.f);
    tampon[thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : makeFloat3(0.f, 0.f, 0.f);
    tampon[(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : makeFloat3(0.f, 0.f, 0.f);
    __syncthreads();
}

//a whole block of 16x16 should into there, x and y corresponds to their real position in the src (or slighly outside)
//at the end, the central 16*16 part of tampon contains the blurred value for each thread
//tampon is of size 32*32
__device__ void GaussianSmart_Device(float3* tampon, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //horizontalBlur on tampon restraint into rectangle [8 - 24][0 - 32] -> 2 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot = 0.;
    float3 out = makeFloat3(0.f, 0.f, 0.f);
    float3 out2 = makeFloat3(0.f, 0.f, 0.f);
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[thy*32 + thx+i]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_x+thx+i >= 0 && tampon_base_x+thx+i < width) tot += gaussiankernel[i];
    }

    //2nd pass in [8 - 24][16 - 32]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out2 += tampon[(thy+16)*32 + thx+i]*gaussiankernel[i];
    }

    __syncthreads();
    tampon[thy*32 + thx+8] = out/tot;
    tampon[(thy+16)*32 + thx+8] = out2/tot;
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 24][8 - 24] -> 1 pass per thread
    out = makeFloat3(0.f, 0.f, 0.f);
    tot = 0.;
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[(thy+i)*32 + thx+8]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_y+thy+i >= 0 && tampon_base_y+thy+i < height) tot += gaussiankernel[i];
    }

    __syncthreads();
    tampon[(thy+8)*32 + thx+8] = out/tot;
    __syncthreads();
}

__device__ void GaussianSmartSharedSave(float3* tampon, float3* dst, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;

    //tampon [8 - 24][8 - 24] -> dst
    if (x >= 0 && x < width && y >= 0 && y < height) dst[y*width + x] = tampon[(thy+8)*32+thx+8];
    __syncthreads();
}

//launch in 16*16
__launch_bounds__(256)
__global__ void GaussianBlur_Kernel(float3* src, float3* dst, int64_t width, int64_t height, float* gaussiankernel){
    int originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly
    for (int scale = 0; scale <= 5; scale++){
        if (originalBl_X < ((width-1)/16+1)*((height-1)/16+1)){
            break;
        } else {
            src += width*height;
            dst += width*height;
            originalBl_X -= ((width-1)/16+1)*((height-1)/16+1);
            width = (width-1)/2+1;
            height = (height-1)/2+1;
        }
    }

    const int blockwidth = (width-1)/16+1;

    const int64_t x = threadIdx.x + 16*(originalBl_X%blockwidth);
    const int64_t y = threadIdx.y + 16*(originalBl_X/blockwidth);

    __shared__ float3 tampon[32*32]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread

    GaussianSmartSharedLoad(tampon, src, x, y, width, height);

    GaussianSmart_Device(tampon, x, y, width, height, gaussiankernel);

    GaussianSmartSharedSave(tampon, dst, x, y, width, height);
}

void gaussianBlur(float3* src, float3* dst, int64_t totalscalesize, int64_t basewidth, int64_t baseheight, float* gaussiankernel_d, hipStream_t stream){
    int64_t w = basewidth;
    int64_t h = baseheight;

    int64_t bl_x = 0;
    for (int scale = 0; scale <= 5; scale++){
        bl_x += ((w-1)/16+1)*((h-1)/16+1);
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    GaussianBlur_Kernel<<<dim3(bl_x), dim3(16, 16), 0, stream>>>(src, dst, basewidth, baseheight, gaussiankernel_d);
    GPU_CHECK(hipGetLastError());
}