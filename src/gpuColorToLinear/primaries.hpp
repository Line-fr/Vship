#pragma once

namespace VshipColorConvert{

//T1 is source, T2 is destination
template<Vship_Primaries_t T1, Vship_Primaries_t T2>
__device__ float3 inline primariesToPrimaries_device(float3 a);

//https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_BT709, Vship_PRIMARIES_INTERNAL>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 0.4124f, fmaf(a.y, 0.3576f, a.z*0.1805f));
    res.y = fmaf(a.x, 0.2126f, fmaf(a.y, 0.7152f, a.z*0.0722f));
    res.z = fmaf(a.x, 0.0193f, fmaf(a.y, 0.1192f, a.z*0.9505f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_BT709>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 3.2410f, fmaf(a.y, -1.5374f, -a.z*0.4986f));
    res.y = fmaf(a.x, -0.9692f, fmaf(a.y, 1.8760f, a.z*0.0416f));
    res.z = fmaf(a.x, 0.0556f, fmaf(a.y, -0.2040f, a.z*1.0570f));
    return res;
}

//https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_BT2020, Vship_PRIMARIES_INTERNAL>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 0.6370f, fmaf(a.y, 0.1446f, a.z*0.1689f));
    res.y = fmaf(a.x, 0.2627f, fmaf(a.y, 0.6780f, a.z*0.0593f));
    res.z = fmaf(a.x, 0.f, fmaf(a.y, 0.0281f, a.z*1.0610f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_BT2020>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 1.71650251f, fmaf(a.y, -0.35558469f, -a.z*0.25337521f));
    res.y = fmaf(a.x, -0.66662561f, fmaf(a.y, 1.61644657f, a.z*0.01577548f));
    res.z = fmaf(a.x, 0.01765521f, fmaf(a.y, -0.0428107f, a.z*0.94208926f));
    return res;
}

//this one is used a lot so we do it in one matrix instead of 2
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_BT2020, Vship_PRIMARIES_BT709>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 1.6605f, fmaf(a.y, -0.5876f, -a.z*0.0728f));
    res.y = fmaf(a.x, -0.1246f, fmaf(a.y, 1.1329f, -a.z*0.0083f));
    res.z = fmaf(a.x, -0.0182f, fmaf(a.y, -0.1006f, a.z*1.1187f));
    return res;
}

//by default, we go through XYZ
template<Vship_Primaries_t T1, Vship_Primaries_t T2>
__device__ float3 inline primariesToPrimaries_device<T1, T2>(float3 a){
    return primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, T2>(primariesToPrimaries_device<T1, Vship_PRIMARIES_INTERNAL>(a));
}

template<Vship_Primaries_t T1, Vship_Primaries_t T2>
__global__ void primariesToPrimaries_kernel(float* p0, float* p1, float* p2, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 val = {p0[x], p1[x], p2[x]};

    val = primariesToPrimaries_device<T1, T2>(val);

    p0[x] = val.x;
    p1[x] = val.y;
    p2[x] = val.z;
}

template<Vship_Primaries_t T1, Vship_Primaries_t T2>
void inline primariesToPrimaries_allTemplate(float* p0, float* p1, float* p2, int64_t width, hipStream_t stream){
    const int thx = 256;
    const int blx = (width + thx -1)/thx;
    primariesToPrimaries_kernel<T1, T2><<<dim3(blx), dim3(thx), 0, stream>>>(p0, p1, p2, width);
}

template<Vship_Primaries_t T2>
void inline primariesToPrimaries_template1(float* p0, float* p1, float* p2, int64_t width, Vship_Primaries_t src_primary, hipStream_t stream){
    switch (src_primary){
        case Vship_PRIMARIES_INTERNAL:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_INTERNAL, T2>(p0, p1, p2, width, stream);
            break;
        case Vship_PRIMARIES_BT709:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_BT709, T2>(p0, p1, p2, width, stream);
            break;
        case Vship_PRIMARIES_BT2020:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_BT2020, T2>(p0, p1, p2, width, stream);
            break;
    }
}

void inline primariesToPrimaries(float* p0, float* p1, float* p2, int64_t width, Vship_Primaries_t src_primary, Vship_Primaries_t dst_primary, hipStream_t stream){
    if (dst_primary == src_primary) return;
    switch (dst_primary){
        case Vship_PRIMARIES_INTERNAL:
            primariesToPrimaries_template1<Vship_PRIMARIES_INTERNAL>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_BT709:
            primariesToPrimaries_template1<Vship_PRIMARIES_BT709>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_BT2020:
            primariesToPrimaries_template1<Vship_PRIMARIES_BT2020>(p0, p1, p2, width, src_primary, stream);
            break;
    }
}

}