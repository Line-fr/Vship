#pragma once

#include "transferToLinear.hpp"

namespace VshipColorConvert{

__device__ float3 inline ncl_yuv_to_rgb_from_kr_kb(float3 val, const float Kr, const float Kb){
    const float Kg = 1.f-Kb-Kr;
    const float uscale = 0.5f/(1.f-Kb);
    const float vscale = 0.5f/(1.f-Kr);

    float3 res;
    res.x = (val.z + val.x * vscale)/(vscale*Kr + 0.5f);
    res.z = (val.y + val.x * uscale)/(uscale*Kb + 0.5f);
    res.y = (val.x - Kr*res.x - Kb*res.z)/Kg;

    return res;
}

template <Vship_YUVMatrix_t matrix>
__device__ float3 inline linearLightYUVConversion(float3 val);

template <Vship_YUVMatrix_t matrix>
__device__ float3 inline gammaLightYUVConversion(float3 val);

//source https://professional.dolby.com/siteassets/pdfs/ictcp_dolbywhitepaper_v071.pdf
//please set the correct transfer metadata to use here (PQ)
template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_BT2100_ICTCP>(float3 val){
    //At that point, we got LMS
    //R = 3.4366066943330784267*L - 2.5064521186562698975*M + 0.069845424323191470954*S
    val.x = fmaf(val.x, 3.4366066943330784267f, fmaf(val.y, -2.5064521186562698975f, val.z*0.069845424323191470954f));
    //G = -0.79132955559892875305*L + 1.9836004517922907339*M -0.19227089619336198096*S
    //or L = 0.290984709320676*R + 0.7293392411733872*M + 0.020323950494063145*S
    //=> G = -0.23026480071281397*R + 1.4064527541936944*M + -0.2083538389058436*S
    val.y = fmaf(val.x, -0.23026480071281397f, fmaf(val.y, 1.4064527541936944f, -val.z*0.2083538389058436f));
    //B = -0.025949899690592673413*L + -0.098913714711726441685*M + 1.1248636144023191151*S
    //or M = 0.16372025297417298*R + 0.7110085973512066*G + 0.14814137075318312*S
    //=> L = 0.4103923143895743*R + 0.5185664708598834*G + 0.12836926542557514*S
    //=> B = -0.02684381778741865*R + -0.08378524945770065*G + 1.106879231557486*S
    val.z = fmaf(val.x, -0.02684381778741865f, fmaf(val.y, -0.08378524945770065f, val.z*1.106879231557486));
    return val;
}

//source https://professional.dolby.com/siteassets/pdfs/ictcp_dolbywhitepaper_v071.pdf
//please set the correct transfer metadata to use here (PQ)
template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_BT2100_ICTCP>(float3 val){
    //at this point, we have ICtCp
    //ICtCp => L'M'S'
    //L' = I + 0.0086090370379327566f* Ct + 0.11102962500302595655f* Cp
    val.x = fmaf(val.y, 0.0086090370379327566f, fmaf(val.z, 0.11102962500302595655f, val.x));
    //M' = I - 0.0086090370379327566f* Ct - 0.11102962500302595655* Cp
    //=> M' = L' - 0.01721807407586551* Ct - 0.22205925000605192* Cp
    val.y = fmaf(val.y, -0.01721807407586551f, fmaf(val.z, -0.22205925000605192f, val.x));
    //S' = I + 0.560031335710679118f* Ct - 0.32062717498731885184f* Cp
    //we have Ct = (L' - M' - 0.22205925000605192* Cp)/0.01721807407586551
    //and I = 0.5L' + 0.5M'
    //=> S' = 33.025782688766114 * L' - 32.025782688766114 * M' - 7.543278084714549 * Cp
    val.z = fmaf(val.x, 33.025782688766114f, fmaf(val.y, -32.025782688766114f, -val.z*7.543278084714549f));

    return val;
}

//source https://en.wikipedia.org/wiki/Rec._2020 (inversed by hand)
template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_BT2020_CL>(float3 val){
    constexpr float Kr = 0.2627f;
    constexpr float Kb = 0.0593f;
    constexpr float Kg = 1.f - Kr - Kb;

    val.y = (val.y - Kr*val.x - Kb*val.z)/Kg;

    return val;
}

//source https://en.wikipedia.org/wiki/Rec._2020 (inverse by hand)
template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_BT2020_CL>(float3 val){
    constexpr float Kr = 0.2627f;
    constexpr float Kb = 0.0593f;
    //constexpr float Kg = 1.f - Kr - Kb;
    //we got YcCbcCrc right now
    const float uscale = 0.5f/(1.f-Kb);
    const float vscale = 0.5f/(1.f-Kr);

    float3 res;
    res.x = (val.z + val.x * vscale)/(vscale*Kr + 0.5f);
    res.z = (val.y + val.x * uscale)/(uscale*Kb + 0.5f);
    res.y = val.x;

    //then we apply the inverse of transfer function to go from Yc'B'R' -> YcBR
    return res;
}

template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_BT709>(float3 val){
    constexpr float Kr = 0.2126f;
    constexpr float Kb = 0.0722f;

    return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
}

template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_BT709>(float3 val){
    return val;
}

//source https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_BT470_BG>(float3 val){
    constexpr float Kr = 0.299f;
    constexpr float Kb = 0.114f;

    return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
}

template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_BT470_BG>(float3 val){
    return val;
}

template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_ST170_M>(float3 val){
    return gammaLightYUVConversion<Vship_MATRIX_BT470_BG>(val);
}

template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_ST170_M>(float3 val){
    return val;
}

//source https://en.wikipedia.org/wiki/Rec._2020 (inverse by hand)
template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_BT2020_NCL>(float3 val){
    constexpr float Kr = 0.2627f;
    constexpr float Kb = 0.0593f;

    return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
}

template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_BT2020_NCL>(float3 val){
    return val;
}

template <>
__device__ float3 inline gammaLightYUVConversion<Vship_MATRIX_RGB>(float3 val){
    return val;
}

template <>
__device__ float3 inline linearLightYUVConversion<Vship_MATRIX_RGB>(float3 val){
    return val;
}

template <Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer>
__device__ float3 inline YUVToLinRGBPipeline_Device(float3 val){
    val = gammaLightYUVConversion<matrix>(val);
    transferLinearize<transfer>(val);
    val = linearLightYUVConversion<matrix>(val);
    return val;
}

template <Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer>
__global__ void YUVToLinRGBPipeline_Kernel(float* p0, float* p1, float* p2, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 val = {p0[x], p1[x], p2[x]};

    val = YUVToLinRGBPipeline_Device<matrix, transfer>(val);
    //if (x == 0) {
    //    float3 tmp = gammaLightYUVConversion<matrix>(float3({p0[x], p1[x], p2[x]}));
    //    printf("at x=%lld, before YUVtoLinRGB: %f %f %f, middle %f %f %f, after: %f %f %f\n", x, p0[x], p1[x], p2[x], tmp.x, tmp.y, tmp.z, val.x, val.y, val.z);
    //}

    p0[x] = val.x;
    p1[x] = val.y;
    p2[x] = val.z;
}

template <Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer>
void YUVToLinRGBPipeline_alltemplate(float* p0, float* p1, float* p2, int64_t width, hipStream_t stream){
    const int thx = 256;
    const int blx = (width + thx -1)/thx;
    YUVToLinRGBPipeline_Kernel<matrix, transfer><<<dim3(blx), dim3(thx), 0, stream>>>(p0, p1, p2, width);
    GPU_CHECK(hipGetLastError());
}

template <Vship_TransferFunction_t transfer>
void YUVToLinRGBPipeline_templateTransfer(float* p0, float* p1, float* p2, int64_t width, Vship_YUVMatrix_t matrix, hipStream_t stream){
    switch (matrix){
        case Vship_MATRIX_RGB:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_RGB, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_BT709:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_BT709, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_BT470_BG:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_BT470_BG, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_ST170_M:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_ST170_M, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_BT2020_NCL:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_BT2020_NCL, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_BT2020_CL:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_BT2020_CL, transfer>(p0, p1, p2, width, stream);
            break;
        case Vship_MATRIX_BT2100_ICTCP:
            YUVToLinRGBPipeline_alltemplate<Vship_MATRIX_BT2100_ICTCP, transfer>(p0, p1, p2, width, stream);
            break;
    }
}

void YUVToLinRGBPipeline(float* p0, float* p1, float* p2, int64_t width, Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer, hipStream_t stream){
    if (transfer == Vship_TRC_Linear && matrix == Vship_MATRIX_RGB) return;
    switch (transfer){
        case Vship_TRC_BT709:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_BT709>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_Linear:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_Linear>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_BT470_BG:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_BT470_BG>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_BT470_M:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_BT470_M>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_BT601:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_BT601>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_ST240_M:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_ST240_M>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_sRGB:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_sRGB>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_PQ:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_PQ>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_ST428:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_ST428>(p0, p1, p2, width, matrix, stream);
            break;
        case Vship_TRC_HLG:
            YUVToLinRGBPipeline_templateTransfer<Vship_TRC_HLG>(p0, p1, p2, width, matrix, stream);
            break;
    }
}

}