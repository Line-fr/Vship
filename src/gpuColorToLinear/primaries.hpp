#pragma once

namespace VshipColorConvert{

//T1 is source, T2 is destination
template<Vship_Primaries_t T1, Vship_Primaries_t T2>
__device__ float3 inline primariesToPrimaries_device(float3 a);

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_INTERNAL>(float3 a){
    return a;
}

//https://fr.mathworks.com/help/images/ref/whitepoint.html
//https://en.wikipedia.org/wiki/NTSC
//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_ST170_M, Vship_PRIMARIES_INTERNAL>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 0.39349893f, fmaf(a.y, 0.3652766f, a.z*0.19162447f));
    res.y = fmaf(a.x, 0.2123645f, fmaf(a.y, 0.70109542f, a.z*0.08654008f));
    res.z = fmaf(a.x, 0.01873804f, fmaf(a.y, 0.1119396f, a.z*0.95812235f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_ST170_M>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 3.5061991f, fmaf(a.y, -1.7398879f, -a.z*0.54408866f));
    res.y = fmaf(a.x, -1.06899334f, fmaf(a.y, 1.97767857f, a.z*0.03516964f));
    res.z = fmaf(a.x, 0.05632201f, fmaf(a.y, -0.1970296f, a.z*1.05023986f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_ST240_M, Vship_PRIMARIES_INTERNAL>(float3 a){
    return primariesToPrimaries_device<Vship_PRIMARIES_ST170_M, Vship_PRIMARIES_INTERNAL>(a);
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_ST240_M>(float3 a){
    return primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_ST170_M>(a);
}

//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
//https://en.wikipedia.org/wiki/PAL
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_BT470_BG, Vship_PRIMARIES_INTERNAL>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 0.4305f, fmaf(a.y, 0.3416f, a.z*0.1783f));
    res.y = fmaf(a.x, 0.2220f, fmaf(a.y, 0.7067f, a.z*0.07132f));
    res.z = fmaf(a.x, 0.0202f, fmaf(a.y, 0.1296f, a.z*0.9391f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_BT470_BG>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 3.064f, fmaf(a.y, -1.3935f, -a.z*0.4758f));
    res.y = fmaf(a.x, -0.9692f, fmaf(a.y, 1.876f, a.z*0.04155f));
    res.z = fmaf(a.x, 0.06788f, fmaf(a.y, -0.2289f, a.z*1.069f));
    return res;
}

//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
//https://en.wikipedia.org/wiki/PAL
template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_BT470_M, Vship_PRIMARIES_INTERNAL>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 0.6068f, fmaf(a.y, 0.1735f, a.z*0.2003f));
    res.y = fmaf(a.x, 0.2989f, fmaf(a.y, 0.5866f, a.z*0.1145f));
    res.z = fmaf(a.x, 0.f, fmaf(a.y, 0.06609f, a.z*1.116f));
    return res;
}

template<>
__device__ float3 inline primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, Vship_PRIMARIES_BT470_M>(float3 a){
    float3 res;
    res.x = fmaf(a.x, 1.910f, fmaf(a.y, -0.5324f, -a.z*0.2882f));
    res.y = fmaf(a.x, -0.9846f, fmaf(a.y, 1.999f, -a.z*0.02831f));
    res.z = fmaf(a.x, 0.05831f, fmaf(a.y, -0.1184f, a.z*0.8976f));
    return res;
}

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
__device__ float3 inline primariesToPrimaries_device2(float3 a){
    if constexpr (T1 == T2){
        return a;
    //special premultiplied matrix
    } else if constexpr (T1 == Vship_PRIMARIES_BT2020 && T2 == Vship_PRIMARIES_BT709){
        return primariesToPrimaries_device<T1, T2>(a);
    } else {
        return primariesToPrimaries_device<Vship_PRIMARIES_INTERNAL, T2>(primariesToPrimaries_device<T1, Vship_PRIMARIES_INTERNAL>(a));
    }
}

template<Vship_Primaries_t T1, Vship_Primaries_t T2>
__global__ void primariesToPrimaries_kernel(float* p0, float* p1, float* p2, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 val = {p0[x], p1[x], p2[x]};

    val = primariesToPrimaries_device2<T1, T2>(val);

    p0[x] = val.x;
    p1[x] = val.y;
    p2[x] = val.z;
}

template<Vship_Primaries_t T1, Vship_Primaries_t T2>
void inline primariesToPrimaries_allTemplate(float* p0, float* p1, float* p2, int64_t width, hipStream_t stream){
    const int thx = 256;
    const int blx = (width + thx -1)/thx;
    primariesToPrimaries_kernel<T1, T2><<<dim3(blx), dim3(thx), 0, stream>>>(p0, p1, p2, width);
    GPU_CHECK(hipGetLastError());
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
        case Vship_PRIMARIES_BT470_BG:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_BT470_BG, T2>(p0, p1, p2, width, stream);
            break;
        case Vship_PRIMARIES_BT470_M:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_BT470_M, T2>(p0, p1, p2, width, stream);
            break;
        case Vship_PRIMARIES_ST170_M:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_ST170_M, T2>(p0, p1, p2, width, stream);
            break;
        case Vship_PRIMARIES_ST240_M:
            primariesToPrimaries_allTemplate<Vship_PRIMARIES_ST240_M, T2>(p0, p1, p2, width, stream);
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
        case Vship_PRIMARIES_BT470_BG:
            primariesToPrimaries_template1<Vship_PRIMARIES_BT470_BG>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_BT470_M:
            primariesToPrimaries_template1<Vship_PRIMARIES_BT470_M>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_ST170_M:
            primariesToPrimaries_template1<Vship_PRIMARIES_ST170_M>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_ST240_M:
            primariesToPrimaries_template1<Vship_PRIMARIES_ST240_M>(p0, p1, p2, width, src_primary, stream);
            break;
        case Vship_PRIMARIES_BT2020:
            primariesToPrimaries_template1<Vship_PRIMARIES_BT2020>(p0, p1, p2, width, src_primary, stream);
            break;
    }
}

}