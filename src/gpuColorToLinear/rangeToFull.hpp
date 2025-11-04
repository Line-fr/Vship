#pragma once

namespace VshipColorConvert {

template<Vship_Range_t T, Vship_Sample_t SampleType>
__device__ float inline FullRange(float a);

//Limited is only taken into account for 8 bit 
template <>
__device__ float inline FullRange<Vship_RangeLimited, Vship_SampleUINT8>(float a){
    return (a-16)/(235-16);
}

template<Vship_Range_t T, Vship_Sample_t SampleType>
__device__ float inline FullRange<T, SampleType>(float a){
    return a;
}

}