namespace ssimu2{

class GaussianHandle{
public:
    float* gaussiankernel_d = NULL;
    float* gaussiankernel_integral_d = NULL;
    void init(){
        float gaussiankernel[4*GAUSSIANSIZE+3];
        gaussiankernel[2*GAUSSIANSIZE+1] = 0; //will be integral
        for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
            gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(TAU*SIGMA*SIGMA));
            gaussiankernel[2*GAUSSIANSIZE+2+i] = gaussiankernel[2*GAUSSIANSIZE+1+i] + gaussiankernel[i];
        }
    
        GPU_CHECK(hipMalloc(&gaussiankernel_d, sizeof(float)*(4*GAUSSIANSIZE+3)));
        GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)gaussiankernel_d, gaussiankernel, (4*GAUSSIANSIZE+3)*sizeof(float)));
        gaussiankernel_integral_d = gaussiankernel_d + 2*GAUSSIANSIZE+1;
    }
    void destroy(){
        GPU_CHECK(hipFree(gaussiankernel_d));
    }
};

__device__ void GaussianSmartSharedLoad(float* tampon, const float* src, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    tampon[thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : 0.f;
    tampon[(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : 0.f;
    tampon[thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : 0.f;
    tampon[(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : 0.f;
    __syncthreads();
}

//a whole block of 16x16 should into there, x and y corresponds to their real position in the src (or slighly outside)
//at the end, the central 16*16 part of tampon contains the blurred value for each thread
//tampon is of size 32*32
__device__ float GaussianSmart_Device(float* tampon, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;

    //horizontalBlur on tampon restraint into rectangle [8 - 24][0 - 32] -> 2 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot;
    float out = 0.f;
    float out2 = 0.f;
    //border handling precompute
    int beg = max((int64_t)0, x-8)-(x-8);
    int end2 = min(width, x+9)-min(width, (x-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[thy*32 + thx+i]*gaussiankernel[i];
        //2nd pass in [8 - 24][16 - 32]
        out2 += tampon[(thy+16)*32 + thx+i]*gaussiankernel[i];
    }

    __syncthreads();
    tampon[thy*32 + thx+8] = out/tot;
    tampon[(thy+16)*32 + thx+8] = out2/tot;
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 24][8 - 24] -> 1 pass per thread
    out = 0.f;
    beg = max((int64_t)0, y-8)-(y-8);
    end2 = min(height, y+9)-min(height, (y-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[(thy+i)*32 + thx+8]*gaussiankernel[i];
    }

    __syncthreads();
    return out/tot;
}

}