#pragma once

namespace cvvdp{

class CubicHermitSplineInterpolator{
    float v1; float v2; float v3; float v4;
public:
    __device__ __host__ CubicHermitSplineInterpolator(const float p0, const float m0, const float p1, const float m1){
        v1 = 2*p0 + m0 - 2*p1 + m1;
        v2 = -3*p0 + 3*p1 - 2*m0 - m1;
        v3 = m0;
        v4 = p0;
    }
    __device__ __host__ float get(const float t){ //cubic uses t between 0 and 1
        float res = v1;
        res *= t;
        res += v2;
        res *= t;
        res += v3;
        res *= t;
        res += v4;
        return res;
    }
};

__global__ void horizontalResizeTranspose_Kernel(float* dst, float* src, int64_t source_width, int64_t source_height, int64_t resize_width){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= resize_width*source_height) return; //x indexed on result

    //destination result
    const int64_t x = thid%resize_width;
    const int64_t y = thid/resize_width;

    //we need to find the closest corresponding left point on original image
    const float approx_source_x = (float)x*(float)source_width/(float)resize_width;
    const int64_t x_oriBase = (int64_t)(approx_source_x);

    const float v1 = src[y*source_width+x_oriBase];
    const float v0 = (x_oriBase-1 >= 0) ? src[y*source_width+x_oriBase-1] : v1;
    const float v2 = (x_oriBase+1 < source_width) ? src[y*source_width+x_oriBase+1] : v1;
    const float v3 = (x_oriBase+2 < source_width) ? src[y*source_width+x_oriBase+2] : v2;
    CubicHermitSplineInterpolator interpolator(v1, (v2-v0)/2.f, v2, (v3-v1)/2.f);
    const float res = interpolator.get(approx_source_x - (float)x_oriBase);

    //dst is transposed
    dst[x*source_height+y] = res;
}

//dest and temp of resize size, src of source size
void resizePlane(float* dest, float* temp, float* src, int64_t source_width, int64_t source_height, int64_t resize_width, int64_t resize_height, hipStream_t stream){
    int th_x = 256;
    int64_t bl1_x = (resize_width*source_height + th_x-1)/th_x;
    int64_t bl2_x = (resize_width*resize_height + th_x-1)/th_x;

    horizontalResizeTranspose_Kernel<<<dim3(bl1_x), dim3(th_x), 0, stream>>>(temp, src, source_width, source_height, resize_width);
    GPU_CHECK(hipGetLastError());
    horizontalResizeTranspose_Kernel<<<dim3(bl2_x), dim3(th_x), 0, stream>>>(dest, temp, source_height, resize_width, resize_height);
    GPU_CHECK(hipGetLastError());
}

}