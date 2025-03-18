namespace ssimu2{

__launch_bounds__(256)
__global__ void downsamplekernel(float3* src, float3* dst, int width, int height){ //threads represents output pixels
    size_t x = threadIdx.x + blockIdx.x*blockDim.x; // < width >> 1 +1
    size_t y = threadIdx.y + blockIdx.y*blockDim.y; // < height >> 1 +1

    int newh = (height-1)/2 + 1;
    int neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    dst[y * neww + x] = src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] *= 0.25f;
    //printf("got %f, %f, %f from %f, %f, %f ; %f, %f, %f ; %f, %f, %f ; %f, %f, %f\n", dst[y * neww + x].x, dst[y * neww + x].y, dst[y * neww + x].z, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x].x, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x].y, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x].z, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x].x, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x].y, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x].z, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x + 1].x, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x + 1].y, src[min((int)(2*y), (int)(newh-1)) * neww + 2*x + 1].z, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x + 1].x, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x + 1].y, src[min((int)(2*y + 1), (int)(newh-1)) * neww + 2*x + 1].z);
}

void inline downsample(float3* src, float3* dst, int width, int height, hipStream_t stream){
    int newh = (height-1)/2 + 1;
    int neww = (width-1)/2 + 1;

    int th_x = std::min(16, neww);
    int th_y = std::min(16, newh);
    int bl_x = (neww-1)/th_x + 1;
    int bl_y = (newh-1)/th_y + 1;
    downsamplekernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(src, dst, width, height);
    GPU_CHECK(hipGetLastError());
}

}