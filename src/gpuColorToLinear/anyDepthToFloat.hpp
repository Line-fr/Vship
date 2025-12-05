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

template<Vship_Sample_t T, bool misalignementHandling>
__device__ float inline PickValue(const uint8_t* const source_plane, const int64_t i, const int64_t stride, const int64_t width){
    constexpr int byteSample = bytesizeSample(T);
    //1 for float types and bitsize for integers
    constexpr int bitSample = bitprecisionSample(T);
    
    const int line = i/width;
    const int column = i%width;
    const uint8_t* baseAdress = source_plane+line*stride + byteSample*column;
    
    //no misalignement possible
    if constexpr (byteSample == 1){
        return *baseAdress;
    }

    if constexpr (misalignementHandling){
        //handle misalignement by byte copy
        uint8_t raw[byteSample];
        #pragma unroll
        for (int i = 0; i < byteSample; i++){
            raw[i] = baseAdress[i];
        }

        if constexpr (T == Vship_SampleFLOAT){
            return *((float*)raw);
        } else if constexpr(T == Vship_SampleHALF){
            return *((__half*)raw);
        } else {
            //only remains byteSample != 1 and non float type => byteSample == 2 uints
            //bitmasking to avoid assuming that garbage data is 0
            return *((uint16_t*)raw) & ((1u << bitSample)-1u);
        }
    } else {
        if constexpr (T == Vship_SampleFLOAT){
            return *((float*)baseAdress);
        } else if constexpr(T == Vship_SampleHALF){
            return *((__half*)baseAdress);
        } else {
            //only remains byteSample != 1 and non float type => byteSample == 2 uints
            //bitmasking to avoid assuming that garbage data is 0
            return *((uint16_t*)baseAdress) & ((1u << bitSample)-1u);
        }
    }
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, bool ColorFamchromaPlane, bool misalignementHandling>
__global__ void convertToFloatPlane_Kernel(float* output_plane, const uint8_t* const source_plane, const int64_t stride, const int64_t width, const int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width*height) return;

    float val = PickValue<SampleType, misalignementHandling>(source_plane, x, stride, width);
    //if (x == 0) printf("raw input val : %f at x = %lld\n", val, x);
    val = FullRange<SampleType, Range, ColorFamchromaPlane>(val);
    //if (x == 0) printf("range adapted input val : %f at x = %lld\n", val, x);
    output_plane[x] = val;
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, bool ColorFamchromaPlane>
__host__ void inline convertToFloatPlaneTemplate(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, hipStream_t stream){
    const int thx = 256;
    const int64_t blx = (width*height + thx -1)/thx;
    constexpr int byteSample = bytesizeSample(SampleType);
    if (stride%byteSample == 0){
        convertToFloatPlane_Kernel<SampleType, Range, ColorFamchromaPlane, false><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
    } else {
        convertToFloatPlane_Kernel<SampleType, Range, ColorFamchromaPlane, true><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
    }
    GPU_CHECK(hipGetLastError());
}

template<Vship_Sample_t SampleType, Vship_Range_t Range, Vship_ColorFamily_t ColorFam>
__host__ void inline convertToFloatPlaneTemplate1(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, bool chromaPlane, hipStream_t stream){
    if (chromaPlane && ColorFam == Vship_ColorYUV){
        return convertToFloatPlaneTemplate<SampleType, Range, true>(output_plane, source_plane, stride, width, height, stream);
    } else {
        return convertToFloatPlaneTemplate<SampleType, Range, false>(output_plane, source_plane, stride, width, height, stream);
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