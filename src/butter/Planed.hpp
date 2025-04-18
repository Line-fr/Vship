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
    void fill0(){
        hipMemsetAsync(mem_d, 0, sizeof(float)*width*height, stream);
    }
    void blur(Plane_d temp, GaussianHandle& gaussianHandle, int i){
        const int gaussiansize = gaussianHandle.getWindow(i);
        float* gaussianKernel = gaussianHandle.get(i);

        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;

        int verticalth_x = 8;
        int verticalth_y = 32;
        int verticalbl_x = (width-1)/verticalth_x+1;
        int verticalbl_y = (height-1)/verticalth_y+1;

        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, mem_d, width, height, gaussianKernel, gaussiansize);
        verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(mem_d, temp.mem_d, width, height, gaussianKernel, gaussiansize);
    }
    void blur(Plane_d dst, Plane_d temp, GaussianHandle& gaussianHandle, int i){
        const int gaussiansize = gaussianHandle.getWindow(i);
        float* gaussianKernel = gaussianHandle.get(i);

        int wh = width*height;

        if (gaussiansize == 8){ //special gaussian blur! It doesnt even use temp
            int th_x = 16;
            int th_y = 16;
            int bl_x = (width-1)/(2*th_x)+1;
            int bl_y = (height-1)/(2*th_y)+1;
            GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst.mem_d, width, height, gaussianKernel);
        } else {
            int th_x = std::min(256, wh);
            int bl_x = (wh-1)/th_x + 1;

            int verticalth_x = 8;
            int verticalth_y = 32;
            int verticalbl_x = (width-1)/verticalth_x+1;
            int verticalbl_y = (height-1)/verticalth_y+1;
            horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, mem_d, width, height, gaussianKernel, gaussiansize);
            verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(dst.mem_d, temp.mem_d, width, height, gaussianKernel, gaussiansize);
        }
    }
    void blurDstNoTemp(Plane_d dst, GaussianHandle& gaussianHandle, int i){
        const int gaussiansize = gaussianHandle.getWindow(i);
        float* gaussianKernel = gaussianHandle.get(i);

        assert(gaussiansize == 8);
        //special gaussian blur! It doesnt even use temp
        int th_x = 16;
        int th_y = 16;
        int bl_x = (width-1)/(2*th_x)+1;
        int bl_y = (height-1)/(2*th_y)+1;
        GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst.mem_d, width, height, gaussianKernel);
    }
    void strideEliminator(float* strided, int stride){
        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        strideEliminator_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, (const uint8_t*)strided, stride, width, height);
    }
    void strideAdder(float* strided, int stride){
        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        strideAdder_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>((const uint8_t*)strided, mem_d, stride, width, height);
    }
    void operator-=(const Plane_d& other){
        subarray(mem_d, other.mem_d, mem_d, width*height, stream);
    }
};

}