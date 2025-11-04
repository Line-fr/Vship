#pragma once

#include "rangeToFull.hpp"

namespace VshipColorConvert{

template<Vship_Sample_t T>
__device__ float inline PickValue(const uint8_t* const source_plane, const int i, const int stride, const int width);

template<>
__device__ float inline PickValue<Vship_SampleFLOAT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((float*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<Vship_SampleHALF>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((__half*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<Vship_SampleUINT8>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint8_t*)(source_plane+line*stride))[column]) / 255.f;
}

template<>
__device__ float inline PickValue<Vship_SampleUINT9>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 511.f;
}

template<>
__device__ float inline PickValue<Vship_SampleUINT10>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 1023.f;
}

template<>
__device__ float inline PickValue<Vship_SampleUINT12>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 4095.f;
}

template<>
__device__ float inline PickValue<Vship_SampleUINT14>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 16383.f;
}

template<>
__device__ float inline PickValue<Vship_SampleUINT16>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 65535.f;
}

template<Vship_Range_t Range, Vship_Sample_t SampleType>
__global__ void convertToFloatPlane_Kernel(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width*height) return;

    float val = PickValue<SampleType>(source_plane, x, stride, width);
    val = FullRange<Range, SampleType>(val);
    output_plane[x] = val;
}

template<Vship_Range_t Range, Vship_Sample_t SampleType>
__host__ void inline convertToFloatPlaneTemplate(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, hipStream_t stream){
    const int thx = 256;
    const int blx = (width*height + thx -1)/thx;
    convertToFloatPlane_Kernel<Range, SampleType><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
}

template<Vship_Range_t Range>
__host__ bool inline convertToFloatPlaneTemplateRange(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Vship_Sample_t T, hipStream_t stream){
    switch (T){
        case Vship_SampleFLOAT:
            convertToFloatPlaneTemplate<Range, Vship_SampleFLOAT>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleHALF:
            convertToFloatPlaneTemplate<Range, Vship_SampleHALF>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT8:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT8>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT9:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT9>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT10:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT10>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT12:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT12>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT14:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT14>(output_plane, source_plane, stride, width, height, stream);
            break;
        case Vship_SampleUINT16:
            convertToFloatPlaneTemplate<Range, Vship_SampleUINT16>(output_plane, source_plane, stride, width, height, stream);
            break;
        default:
            return 1;
    }
    return 0;
}

//it eliminates stride as well
__host__ bool inline convertToFloatPlane(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Vship_Sample_t T, Vship_Range_t Range, hipStream_t stream){
    switch (Range){
        case Vship_RangeFull:
            return convertToFloatPlaneTemplateRange<Vship_RangeFull>(output_plane, source_plane, stride, width, height, T, stream);
        case Vship_RangeLimited:
            return convertToFloatPlaneTemplateRange<Vship_RangeLimited>(output_plane, source_plane, stride, width, height, T, stream);
        default:
            return 1;
    }
}

}