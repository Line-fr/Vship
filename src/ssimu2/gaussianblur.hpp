__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = width*height;
    int w = width; int h = height;
    for (int i = 0; i < 5; i++){
        if (x < size) break; //we have the right scale
        w = (w-1)/2+1; h = (h-1)/2+1;
        src += size; dst += size; x -= size;
        size = h*w;
    }
    if (x >= size) return;

    int current_line = x/w;

    float tot = 0;
    float3 out; out.x = 0; out.y = 0; out.z = 0;
    for (int i = max(x-GAUSSIANSIZE, current_line*w); i <= min(x+GAUSSIANSIZE, (current_line+1)*w-1); i++){
        out += src[i]*gaussiankernel[GAUSSIANSIZE+i-x];
        tot += gaussiankernel[GAUSSIANSIZE+i-x];
        //printf("%f at %d\n", gaussiankernel[GAUSSIANSIZE+i-x], GAUSSIANSIZE+i-x);
    }
    out /= tot;
    dst[x] = out;
}

__launch_bounds__(256)
__global__ void verticalBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = width*height;
    int w = width; int h = height;
    for (int i = 0; i < 5; i++){
        if (x < size) break; //we have the right scale
        w = (w-1)/2+1; h = (h-1)/2+1;
        src += size; dst += size; x -= size;
        size = h*w;
    }
    if (x >= size) return;
    int current_line = x/w;
    int current_column = x%w;

    float tot = 0.0f;
    float3 out; out.x = out.y = out.z = 0.0f;
    for (int i = max(current_line-GAUSSIANSIZE, 0); i <= min(current_line+GAUSSIANSIZE, h-1); i++){
        out += src[i * w + current_column]*gaussiankernel[GAUSSIANSIZE+i-current_line];
        tot += gaussiankernel[GAUSSIANSIZE+i-current_line];
        //printf("%f\n", gaussiankernel[GAUSSIANSIZE+i-current_line]);
    }
    out /= tot;
    dst[x] = out;
    //printf("from %f, %f, %f to %f, %f, %f\n", src[x].x, src[x].y, src[x].z, dst[x].x, dst[x].y, dst[x].z);
}

void gaussianBlur(float3* src, float3* dst, float3* temp, int totalscalesize, int basewidth, int baseheight, float* gaussiankernel_d, hipStream_t stream){
    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;

    th_x = std::min(256, totalscalesize);
    bl_x = (totalscalesize-1)/th_x + 1;
    horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src, temp, w, h, gaussiankernel_d);
    GPU_CHECK(hipGetLastError());
    verticalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, dst, w, h, gaussiankernel_d);
    GPU_CHECK(hipGetLastError());
}