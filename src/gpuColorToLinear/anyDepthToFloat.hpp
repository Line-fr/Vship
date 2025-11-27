#pragma once

#include "rangeToFull.hpp"

namespace VshipColorConvert{

__device__ __host__ constexpr int bytesizeSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_SampleUINT8:
            return 1;
        case Vship_SampleUINT9:
        case Vship_SampleUINT10:
        case Vship_SampleUINT12:
        case Vship_SampleUINT14:
        case Vship_SampleUINT16:
        case Vship_SampleHALF:
            return 2;
        case Vship_SampleFLOAT:
            return 4;
    }
    return 0;
}

template<Vship_Sample_t T>
__device__ float inline PickValue(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width);

template<>
__device__ float inline PickValue<Vship_SampleFLOAT>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return ((float*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<Vship_SampleHALF>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return ((__half*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<Vship_SampleUINT8>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint8_t*)(source_plane+line*stride))[column]);
}

template<>
__device__ float inline PickValue<Vship_SampleUINT9>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]);
}

template<>
__device__ float inline PickValue<Vship_SampleUINT10>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    //if (i == 0) printf("PickValue level: %u\n", ((uint16_t*)(source_plane+line*stride))[column]);
    return (float)(((uint16_t*)(source_plane+line*stride))[column]);
}

template<>
__device__ float inline PickValue<Vship_SampleUINT12>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]);
}

template<>
__device__ float inline PickValue<Vship_SampleUINT14>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]);
}

template<>
__device__ float inline PickValue<Vship_SampleUINT16>(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]);
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, Vship_ColorFamily_t ColorFam, bool chromaPlane>
__global__ void convertToFloatPlane_Kernel(float* output_plane, const uint8_t* const source_plane, const int64_t stride, const int64_t width, const int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width*height) return;

    float val = PickValue<SampleType>(source_plane, x, stride, width);
    //if (x == 0) printf("raw input val : %f at x = %lld\n", val, x);
    val = FullRange<SampleType, Range, ColorFam, chromaPlane>(val);
    //if (x == 0) printf("range adapted input val : %f at x = %lld\n", val, x);
    output_plane[x] = val;
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, Vship_ColorFamily_t ColorFam, bool chromaPlane>
__host__ void inline convertToFloatPlaneTemplate(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, hipStream_t stream){
    const int thx = 256;
    const int64_t blx = (width*height + thx -1)/thx;
    convertToFloatPlane_Kernel<SampleType, Range, ColorFam, chromaPlane><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, Vship_ColorFamily_t ColorFam>
__host__ void inline convertToFloatPlaneTemplate1(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, bool chromaPlane, hipStream_t stream){
    if (chromaPlane){
        return convertToFloatPlaneTemplate<SampleType, Range, ColorFam, true>(output_plane, source_plane, stride, width, height, stream);
    } else {
        return convertToFloatPlaneTemplate<SampleType, Range, ColorFam, false>(output_plane, source_plane, stride, width, height, stream);
    }
}

template<Vship_Sample_t SampleType, Vship_Range_t Range>
__host__ void inline convertToFloatPlaneTemplate2(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Vship_ColorFamily_t colorFam, bool chromaPlane, hipStream_t stream){
    switch (colorFam){
        case Vship_ColorRGB:
            return convertToFloatPlaneTemplate1<SampleType, Range, Vship_ColorRGB>(output_plane, source_plane, stride, width, height, chromaPlane, stream);
        case Vship_ColorYUV:
            return convertToFloatPlaneTemplate1<SampleType, Range, Vship_ColorYUV>(output_plane, source_plane, stride, width, height, chromaPlane, stream);
    }
}

template<Vship_Sample_t SampleType>
__host__ void inline convertToFloatPlaneTemplate3(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Vship_Range_t range, Vship_ColorFamily_t colorFam, bool chromaPlane, hipStream_t stream){
    switch (range){
        case Vship_RangeFull:
            return convertToFloatPlaneTemplate2<SampleType, Vship_RangeFull>(output_plane, source_plane, stride, width, height, colorFam, chromaPlane, stream);
        case Vship_RangeLimited:
            return convertToFloatPlaneTemplate2<SampleType, Vship_RangeLimited>(output_plane, source_plane, stride, width, height, colorFam, chromaPlane, stream);
    }
}

//it eliminates stride as well
__host__ void inline convertToFloatPlane(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Vship_Sample_t sampleType, Vship_Range_t range, Vship_ColorFamily_t colorFam, bool chromaPlane, hipStream_t stream){
    switch (sampleType){
        case Vship_SampleFLOAT:
            return convertToFloatPlaneTemplate3<Vship_SampleFLOAT>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleHALF:
            return convertToFloatPlaneTemplate3<Vship_SampleHALF>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT8:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT8>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT9:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT9>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT10:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT10>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT12:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT12>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT14:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT14>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
        case Vship_SampleUINT16:
            return convertToFloatPlaneTemplate3<Vship_SampleUINT16>(output_plane, source_plane, stride, width, height, range, colorFam, chromaPlane, stream);
    }
}

}