#pragma once

namespace VshipColorConvert {

template<Vship_Range_t T, Vship_ColorFamily_t ColorFam, bool chromaPlane>
__device__ float inline FullRange(float a);

//Limited has the same formula for all bitdepth
template <>
__device__ float inline FullRange<Vship_RangeLimited, Vship_ColorYUV, false>(float a){
    return (a*255.f-16.f)/(235.f-16.f);
}

//chroma goes from 16 to 240
//Limited has the same formula for all bitdepth
template <>
__device__ float inline FullRange<Vship_RangeLimited, Vship_ColorYUV, true>(float a){
    return (a*255.f-128.f)/(240.f-16.f); //integrated -0.5f
}

template<>
__device__ float inline FullRange<Vship_RangeFull, Vship_ColorYUV, false>(float a){
    return a;
}

template<>
__device__ float inline FullRange<Vship_RangeFull, Vship_ColorYUV, true>(float a){
    return a - 0.5f;
}

template<>
__device__ float inline FullRange<Vship_RangeLimited, Vship_ColorRGB, true>(float a){
    return (a*255.f-16.f)/(235.f-16.f);;
}

template<>
__device__ float inline FullRange<Vship_RangeLimited, Vship_ColorRGB, false>(float a){
    return (a*255.f-16.f)/(235.f-16.f);;
}

template<>
__device__ float inline FullRange<Vship_RangeFull, Vship_ColorRGB, true>(float a){
    return a;
}

template<>
__device__ float inline FullRange<Vship_RangeFull, Vship_ColorRGB, false>(float a){
    return a;
}

}