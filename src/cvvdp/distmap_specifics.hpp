#pragma once

namespace cvvdp{

__device__ __host__ float toJOD(float a){
    if (a > 0.1f){
        return 10.f - jod_a * powf(a, jod_exp);
    } else {
        const float jod_a_p = jod_a * (powf(0.1f, jod_exp-1.f));
        return 10.f - jod_a_p * a;
    }
}

__device__ float inline topow4(float el){
    float res = el;
    res *= res;
    res *= res;
    return res;
}

template<int divisor>
__global__ void mergeChromaNorm_Kernel(float* p0, float* p1, float* p2, float* p3, int64_t size){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= size) return;

    p0[thid] = powf((topow4(p0[thid]) + topow4(p1[thid]) + topow4(p2[thid]) + topow4(p3[thid]))*0.25, 0.25f)/(float)divisor;
}

__global__ void JODIZE_Kernel(float* p, int64_t size){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= size) return;

    p[thid] = 1.f - (toJOD(p[thid]))*0.1f;
}

template<int divisor>
void mergeChromaNorm(float* p0, float* p1, float* p2, float* p3, int64_t size, hipStream_t stream){
    int th_x = 256;
    int bl_x = (size+th_x-1)/th_x;
    mergeChromaNorm_Kernel<divisor><<<dim3(bl_x), dim3(th_x), 0, stream>>>(p0, p1, p2, p3, size);
    GPU_CHECK(hipGetLastError());
}

//we should have D values in Lpyr bands and channels.
//Lbkg planes will serve as a buffer and the final result willl be in Lbkg band 0
void getDistMap(LpyrManager& Lpyr, hipStream_t stream){
    //first step is merging chroma planes
    for (int band = 0; band < Lpyr.getSize(); band++){
        const auto [w, h] = Lpyr.getResolution(band);
        if (band == 0 || band == Lpyr.getSize()-1){
            mergeChromaNorm<1>(Lpyr.getContrast(0, band), Lpyr.getContrast(1, band), Lpyr.getContrast(2, band), Lpyr.getContrast(3, band), w*h, stream);
        } else {
            mergeChromaNorm<2>(Lpyr.getContrast(0, band), Lpyr.getContrast(1, band), Lpyr.getContrast(2, band), Lpyr.getContrast(3, band), w*h, stream);
        }
    }
    //reconstruction
    for (int band = Lpyr.getSize()-2; band >= 0; band--){
        const auto [w, h] = Lpyr.getResolution(band);
        gaussPyrExpand<false, true, false>(Lpyr.getContrast(0, band), Lpyr.getContrast(0, band+1), w, h, stream);
    }
    const auto [w0, h0] = Lpyr.getResolution(0);
    GPU_CHECK(hipMemcpyDtoDAsync(Lpyr.getLbkg(0), Lpyr.getContrast(0, 0), sizeof(float)*w0*h0, stream));
    JODIZE_Kernel<<<dim3((w0*h0+255)/256), dim3(256), 0, stream>>>(Lpyr.getLbkg(0), w0*h0);
    GPU_CHECK(hipGetLastError());
}

}