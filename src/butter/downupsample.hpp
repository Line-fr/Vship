namespace butter{

__launch_bounds__(256)
__global__ void downsamplekernel(float* src, float* dst, int width, int height){ //threads represents output pixels
    size_t x = threadIdx.x + blockIdx.x*blockDim.x; // < width >> 1 +1
    size_t y = threadIdx.y + blockIdx.y*blockDim.y; // < height >> 1 +1

    int newh = (height-1)/2 + 1;
    int neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    dst[y * neww + x] = 0.0f;
    dst[y * neww + x] += src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] *= 0.25f;
    //if ((y*2+1)*width + x*2 == 10000) printf("got %f\n", dst[y*neww + x]);
}

void inline downsample(float* src, float* dst, int width, int height, hipStream_t stream){
    int newh = (height-1)/2 + 1;
    int neww = (width-1)/2 + 1;

    int th_x = std::min(16, neww);
    int th_y = std::min(16, newh);
    int bl_x = (neww-1)/th_x + 1;
    int bl_y = (newh-1)/th_y + 1;
    downsamplekernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(src, dst, width, height);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void addsupersample2X_kernel(float* diffmap, float* diffmapsmall, int width, int height, float w){ //threads represents output pixels
    size_t x = threadIdx.x + blockIdx.x*blockDim.x; // < width >> 1 +1
    size_t y = threadIdx.y + blockIdx.y*blockDim.y; // < height >> 1 +1

    //int newh = (height-1)/2 + 1;
    int neww = (width-1)/2 + 1;
    
    if (x >= width || y >= height) return;

    //if (y*width+x == 1841628) printf("small got %f and normal got %f\n", diffmapsmall[(y/2)*neww+x/2], diffmap[y*width+x]);

    diffmap[y*width+x] *= 1.0 - 0.3*w;
    diffmap[y*width+x] += w*diffmapsmall[(y/2)*neww+x/2];
}

void inline addsupersample2X(float* diffmap, float* diffmapsmall, int width, int height, float w, hipStream_t stream){
    int th_x = std::min(16, width);
    int th_y = std::min(16, height);
    int bl_x = (width-1)/th_x + 1;
    int bl_y = (height-1)/th_y + 1;
    addsupersample2X_kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(diffmap, diffmapsmall, width, height, w);
}

}