#pragma once

namespace cvvdp{

//reverse BT709 transfer on GPU
__device__ inline void rgb_to_linrgbfunc(float& a){
    if (a < 0){
        a = -powf(-a, 2.4);
    } else {
        a = powf(a, 2.4);
    }
}

__device__ inline void rgb_to_linrgb(float3& a){
    rgb_to_linrgbfunc(a.x);
    rgb_to_linrgbfunc(a.y);
    rgb_to_linrgbfunc(a.z);
}

//for BT709 primaries
__device__ inline void linrgb_to_xyz(float3& a){
    float3 out;

    out.x = fmaf(0.4124564, a.x, fmaf(0.3575761, a.y, 0.1804375*a.z));
    out.y = fmaf(0.2126729, a.x, fmaf(0.7151522, a.y, 0.0721750*a.z));
    out.z = fmaf(0.0193339, a.x, fmaf(0.1191920, a.y, 0.9503041*a.z));

    a = out;
}

__device__ inline void xyz_to_LMS2006(float3& a){
    float3 out;

    out.x = fmaf(0.187596268556126, a.x, fmaf(0.585168649077728, a.y, -0.026384263306304*a.z));
    out.y = fmaf(-0.133397430663221, a.x, fmaf(0.405505777260049, a.y, 0.034502127690364*a.z));
    out.z = fmaf(0.000244379021663, a.x, fmaf(-0.000542995890619, a.y, 0.019406849066323*a.z));

    a = out;
}

__device__ inline void LMS2006_to_DKLd65(float3& a){
    //a = x+y
    a.x = a.x + a.y;
    //b = x -2.311130179947035y = a -1.311130179947035y
    a.y = a.x - 1.311130179947035*a.y;
    //c = -x-y + 50.977571328718781z = -a + 50.977571328718781z
    a.z = 50.977571328718781*a.z - a.x;
}

//not used but exist in cvvdp in case of parameter change
__global__ void rgb_to_loglms_dklKernel(float* p1, float* p2, float* p3, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 src = {p1[x], p2[x], p3[x]};

    rgb_to_linrgb(src);
    linrgb_to_xyz(src);
    xyz_to_LMS2006(src);
    src.x = log10(src.x); src.y = log10(src.y); src.z = log10(src.z);
    LMS2006_to_DKLd65(src);

    p1[x] = src.x; p2[x] = src.y; p3[x] = src.z;
}


__global__ void rgb_to_dklKernel(float* p1, float* p2, float* p3, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 src = {p1[x], p2[x], p3[x]};

    rgb_to_linrgb(src);
    linrgb_to_xyz(src);
    xyz_to_LMS2006(src);
    LMS2006_to_DKLd65(src);

    p1[x] = src.x; p2[x] = src.y; p3[x] = src.z;
}

void inline rgb_to_dkl(float* src_d[3], int64_t width, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (width+th_x-1)/th_x;
    rgb_to_dklKernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], width);
}


}