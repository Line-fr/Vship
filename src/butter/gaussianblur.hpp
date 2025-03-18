namespace butter{

__global__ void loadGaussianKernel(float* gaussiankernel, int gaussiansize, float sigma){
    const int x = threadIdx.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x] = expf(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
}

//transpose the result at the end
__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float* dst, float* src, int w, int h, float border_ratio, float weight_no_border, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(x-gaussiansize, current_line*w); i <= min(x+gaussiansize, (current_line+1)*w-1); i++){
        const float gauss = gaussiankernel[gaussiansize+i-x];
        out += src[i]*gauss;
        weight += gauss;
    }
    dst[x] = out*(1.0f/((1.0f-border_ratio)*weight + border_ratio*weight_no_border));
}

__launch_bounds__(256)
__global__ void verticalBlur_Kernel(float* dst, float* src, int w, int h, float border_ratio, float weight_no_border, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(current_line-gaussiansize, 0); i <= min(current_line+gaussiansize, h-1); i++){
        out += src[i * w + current_column]*gaussiankernel[gaussiansize+i-current_line];
        weight += gaussiankernel[gaussiansize+i-current_line];
        //if (threadIdx.x == 0) printf("%f at %d\n", gaussiankernel[gaussiansize+i-current_line], gaussiansize+i-current_line);
    }
    float resweight = (1.0f-border_ratio)*weight + border_ratio*weight_no_border;
    dst[x] = out/resweight;
    //printf("from %f to %f with resweight %f, weight %f, border_ratio %f, weight_no_border %f\n", src[x], dst[x], resweight, weight, border_ratio, weight_no_border);
}

}