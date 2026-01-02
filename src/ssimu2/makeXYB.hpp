#ifndef MAKEXYBHPP
#define MAKEXYBHPP

namespace ssimu2{

__device__ inline void opsin_absorbance(float3& a){
    float3 out;
    const float opsin_bias = 0.0037930734f;
    out.x = fmaf(0.30f, a.x,
    fmaf(0.622f, a.y,
    fmaf(0.078f, a.z,
    opsin_bias)));

    out.y = fmaf(0.23f, a.x,
    fmaf(0.692f, a.y,
    fmaf(0.078f, a.z,
    opsin_bias)));

    out.z = fmaf(0.24342269f, a.x,
    fmaf(0.20476745f, a.y,
    fmaf(0.55180986f, a.z,
    opsin_bias)));

    a = out;
}

__device__ inline void mixed_to_xyb(float3& a){
    a.x = 0.5f * (a.x - a.y);
    a.y = a.x + a.y;
}

__device__ inline void linear_rgb_to_xyb(float3& a){
    const float abs_bias = -0.1559542025327239f;
    opsin_absorbance(a);
    //printf("from %f to %f\n", a.x, cbrtf(a.x*((int)(a.x >= 0))));
    a.x = cbrtf(a.x * ((int)(a.x >= 0))) + abs_bias;
    a.y = cbrtf(a.y * ((int)(a.y >= 0))) + abs_bias;
    a.z = cbrtf(a.z * ((int)(a.z >= 0))) + abs_bias;
    //printf("got %f, %f, %f\n", a.x, a.y, a.z);
    mixed_to_xyb(a);
}

__device__ inline void make_positive_xyb(float3& a) {
    a.z = (a.z - a.y) + 0.55;
    a.x = a.x * 14.0f + 0.42;  
    a.y += 0.01;
}

__device__ inline void rgb_to_positive_xyb_d(float3& a){
    linear_rgb_to_xyb(a);
    make_positive_xyb(a);
}

__launch_bounds__(256)
__global__ void rgb_to_positive_xyb_Kernel(float* p0, float* p1, float* p2, int64_t width){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width) return;
    float3 packed = {p0[x], p1[x], p2[x]};
    rgb_to_positive_xyb_d(packed);
    p0[x] = packed.x;
    p1[x] = packed.y;
    p2[x] = packed.z;
    //if (x == 10000) printf("from %f, %f, %f to %f, %f, %f\n", old.x, old.y, old.z, array[x].x, array[x].y, array[x].z);
}

__host__ inline void rgb_to_positive_xyb(float* src_d[3], int64_t width, hipStream_t stream){
    int64_t th_x = std::min((int64_t)256, width);
    int64_t bl_x = (width-1)/th_x + 1;
    rgb_to_positive_xyb_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], width);
    GPU_CHECK(hipGetLastError());
}

}

#endif