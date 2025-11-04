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

//for transfers that act on single components
template <Vship_TransferFunction_t TRC>
__device__ void inline transferLinearize(float& a);

template <Vship_TransferFunction_t TRC>
__device__ void inline transferLinearize(float3& a);

//https://github.com/gfxdisp/ColorVideoVDP/blob/main/pycvvdp/display_model.py
//Note: this is HLG
template<>
__device__ void inline transferLinearize<Vship_TRC_HLG>(float3& val){
    constexpr float a = 0.17883277f;
    constexpr float b = 1.f - 4.f * a;
    //constexpr float c = 0.5f - a * logf(4*a); C++ doesnt accept constexpr logf
    constexpr float c = 0.5f - a * -0.3350097945111627f;
    constexpr float gamma = 1.2;

    //inverse OETF
    if (val.x <= 0.5f){
        val.x = val.x*val.x / 3.f;
    } else {
        val.x = (expf((val.x - c)/a)+b)/12.f;
    }
    if (val.y <= 0.5f){
        val.y = val.y*val.y / 3.f;
    } else {
        val.y = (expf((val.y - c)/a)+b)/12.f;
    }
    if (val.z <= 0.5f){
        val.z = val.z*val.z / 3.f;
    } else {
        val.z = (expf((val.z - c)/a)+b)/12.f;
    }

    //OOTF
    const float Ys = 0.2627f*val.x + 0.6780f*val.y + 0.0593*val.z;
    val = val * powf(Ys, gamma-1.f);
}

//apply linear on all 3 components if it was defined on 1
template <Vship_TransferFunction_t TRC>
__device__ void inline transferLinearize<TRC>(float3& a){
    transferLinearize<TRC>(a.x);
    transferLinearize<TRC>(a.y);
    transferLinearize<TRC>(a.z);
}


//define transferLinearize

template <>
__device__ void inline transferLinearize<Vship_TRC_Linear>(float& a){
}

//source Wikipedia
template <>
__device__ void inline transferLinearize<Vship_TRC_sRGB>(float& a){
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
__device__ void inline transferLinearize<Vship_TRC_BT709>(float& a){
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

/*
template <>
__device__ void inline transferLinearize<AVCOL_TRC_GAMMA22>(float& a){
    gamma_to_linrgbfunc(a, 2.2f);
}

template <>
__device__ void inline transferLinearize<AVCOL_TRC_GAMMA28>(float& a){
    gamma_to_linrgbfunc(a, 2.8f);
}*/

//source https://github.com/haasn/libplacebo/blob/master/src/shaders/colorspace.c (14/05/2025 line 670)
template <>
__device__ void inline transferLinearize<Vship_TRC_ST428>(float& a){
    gamma_to_linrgbfunc(a, 2.6f);
    a *= 52.37/48.;
}

//source https://fr.wikipedia.org/wiki/Perceptual_Quantizer
//Note: this is PQ
template<>
__device__ void inline transferLinearize<Vship_TRC_PQ>(float& a){
    const float c1 = 107./128.;
    const float c2 = 2413./128.;
    const float c3 = 2392./128.;
    a = powf(a, 32./2523.);
    a = fmaxf(a - c1, 0.f)/(c2 - c3*a);
    a = powf(a, 8192./1305.);
    a *= 10000;
}

}