#pragma once

/*
List of currently implemented transfer functions

Linear
sRGB
BT709
GAMMA22
GAMMA28
ST428
PQ
(HLG)
*/

namespace VshipColorConvert{

template <AVColorTransferCharacteristic TRANSFER_TYPE>
__device__ void inline transferLinearize(float& a);

//apply linear on all 3 components
template <AVColorTransferCharacteristic TRANSFER_TYPE>
__device__ void inline transferLinearize(float3& a){
    transferLinearize<TRANSFER_TYPE>(a.x);
    transferLinearize<TRANSFER_TYPE>(a.y);
    transferLinearize<TRANSFER_TYPE>(a.z);
}


//define transferLinearize

template <>
__device__ void inline transferLinearize<AVCOL_TRC_LINEAR>(float& a){
}

//source Wikipedia
template <>
__device__ void inline transferLinearize<AVCOL_TRC_IEC61966_2_1>(float& a){
    if (a < 0){
        if (a < -0.04045f){
            a = -powf(((-a+0.055)*(1.0/1.055)), 2.4f);
        } else {
            a *= 1.0/12.92;
        }
    } else {
        if (a > 0.04045f){
            a = powf(((a+0.055)*(1.0/1.055)), 2.4f);
        } else {
            a *= 1.0/12.92;
        }
    }
}

//source https://www.image-engineering.de/library/technotes/714-color-spaces-rec-709-vs-srgb
//I inversed the function myself
/*
template <>
__device__ void inline transferLinearize<AVCOL_TRC_BT709>(float& a){
    if (a < 0){
        if (a < -0.081f){
            a = -powf(((-a+0.099)/1.099), 2.2f);
        } else {
            a *= 1.0/4.5;
        }
    } else {
        if (a > 0.081f){
            a = powf(((a+0.099)/1.099), 2.2f);
        } else {
            a *= 1.0/4.5;
        }
    }
}*/

//BT709 as Pure gamma since it is what is commonly used in reality
template <>
__device__ void inline transferLinearize<AVCOL_TRC_BT709>(float& a){
    if (a < 0){
        a = -powf(-a, 2.4);
    } else {
        a = powf(a, 2.4);
    }
}

__device__ inline void gamma_to_linrgbfunc(float& a, float gamma){
    if (a < 0){
        a = -powf(-a, gamma);
    } else {
        a = powf(a, gamma);
    }
}

template <>
__device__ void inline transferLinearize<AVCOL_TRC_GAMMA22>(float& a){
    gamma_to_linrgbfunc(a, 2.2f);
}

template <>
__device__ void inline transferLinearize<AVCOL_TRC_GAMMA28>(float& a){
    gamma_to_linrgbfunc(a, 2.8f);
}

//source https://github.com/haasn/libplacebo/blob/master/src/shaders/colorspace.c (14/05/2025 line 670)
template <>
__device__ void inline transferLinearize<AVCOL_TRC_SMPTE428>(float& a){
    gamma_to_linrgbfunc(a, 2.6f);
    a *= 52.37/48.;
}

//source https://fr.wikipedia.org/wiki/Perceptual_Quantizer
//Note: this is PQ
template<>
__device__ void inline transferLinearize<AVCOL_TRC_SMPTE2084>(float& a){
    const float c1 = 107./128.;
    const float c2 = 2413./128.;
    const float c3 = 2392./128.;
    a = powf(a, 32./2523.);
    a = fmaxf(a - c1, 0.f)/(c2 - c3*a);
    a = powf(a, 8192./1305.);
    a *= 10000;
}

/*
//https://en.wikipedia.org/wiki/Hybrid_log%E2%80%93gamma
//Note: this is HLG
template<>
__device__ void inline transferLinearize<AVCOL_TRC_ARIB_STD_B67>(float& a){

}
*/

}