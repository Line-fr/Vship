namespace butter {

class Plane_d{
public:
    int width, height;
    float* mem_d; //must be of size >= sizeof(float)*width*height;
    hipStream_t stream;
    Plane_d(float* mem_d, int width, int height, hipStream_t stream){
        this->mem_d = mem_d;
        this->height = height;
        this->width = width;
        this->stream = stream;
    }
    Plane_d(float* mem_d, Plane_d src){
        this->mem_d = mem_d;
        width = src.width;
        height = src.height;
        stream = src.stream;
        hipMemcpyDtoDAsync(mem_d, src.mem_d, sizeof(float)*width*height, stream);
    }
    void blur(Plane_d temp, float sigma, float border_ratio, float* gaussiankernel){
        const int gaussiansize = (int)(sigma * 5);
        loadGaussianKernel<<<dim3(1), dim3(2*gaussiansize+1), 0, stream>>>(gaussiankernel, gaussiansize, sigma);

        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        float weight_no_border = 0;
        for (int i = 0; i < 2*gaussiansize+1; i++){
            weight_no_border += gaussiankernel[i];
        }
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, temp.mem_d, width, height, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, mem_d, height, width, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
    }
    void blur(Plane_d dst, Plane_d temp, float sigma, float border_ratio, float* gaussiankernel){
        const int gaussiansize = (int)(sigma * 5);
        loadGaussianKernel<<<dim3(1), dim3(2*gaussiansize+1), 0, stream>>>(gaussiankernel, gaussiansize, sigma);

        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        float weight_no_border = 0;
        for (int i = 0; i < 2*gaussiansize+1; i++){
            weight_no_border += gaussiankernel[i];
        }
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, temp.mem_d, width, height, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, dst.mem_d, height, width, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
    }
    void strideEliminator(float* strided, int stride){
        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        strideEliminator_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, strided, stride, width, height);
    }
};

}