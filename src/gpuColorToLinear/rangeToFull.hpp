#pragma once

namespace VshipColorConvert {

__device__ __host__ constexpr int bitprecisionSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_SampleUINT8:
            return 8;
        case Vship_SampleUINT9:
            return 9;
        case Vship_SampleUINT10:
            return 10;
        case Vship_SampleUINT12:
            return 12;
        case Vship_SampleUINT14:
            return 14;
        case Vship_SampleUINT16:
            return 16;
        default:
            return 1;
    }
}

template<Vship_Sample_t sampleType, Vship_Range_t Range, bool ColorFamchromaPlane>
__device__ float inline FullRange(float a){
    constexpr int bitdepth = bitprecisionSample(sampleType);
    if constexpr (Range == Vship_RangeFull){
        constexpr float normalization = (1 << bitdepth) - 1; //float and half get 1
        //put in range [0, 1]
        a /= normalization;
        //if UV -> -0.5
        if constexpr (ColorFamchromaPlane) a -= 0.5f;
        return a;
    } else {
        //limited
        if constexpr (bitdepth >= 8){
            //float or half have bitdepth 1. Here we have all uint types
            //-> range [0, 256[
            a /= (1 << (bitdepth - 8));
        } else {
            //float or half have bitdepth 1.
            //-> range [0, 256]
            //these types should ideally not use limited range since the formula is incorrect.
            a *= 256;
        }
        //range [0, 256] BUT we are in limited
        if constexpr (ColorFamchromaPlane){
            //chroma YUV formula
            return (a-128.f)/224.f;
        } else {
            //luma formula
            return (a-16.f)/219.f;
        }
    }
}

}