#pragma once

namespace VshipColorConvert{

template<AVColorPrimaries T>
__device__ void inline PrimariesToBT709(float3& a);

template<>
__device__ void inline PrimariesToBT709<AVCOL_PRI_BT709>(float3& a){

}

//https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf
template<>
__device__ void inline PrimariesToBT709<AVCOL_PRI_BT2020>(float3& a){
    float3 out;
    out.x = fmaf(1.6605f, a.x, fmaf(-0.5876f, a.y, -a.z*0.0728f));
    out.y = fmaf(-0.1246f, a.x, fmaf(1.1329f, a.y, -a.z*0.0083f));
    out.z = fmaf(-0.0182f, a.x, fmaf(-0.1006f, a.y, a.z*1.1187f));
    a = out;
}

}