#pragma once

namespace VshipColorConvert{

enum Sample_Type{
    COLOR_FLOAT,
    COLOR_HALF,
    COLOR_8BIT,
    COLOR_9BIT,
    COLOR_10BIT,
    COLOR_12BIT,
    COLOR_14BIT,
    COLOR_16BIT,
};

template<Sample_Type T>
__device__ float inline PickValue(const uint8_t* const source_plane, const int i, const int stride, const int width);

template<>
__device__ float inline PickValue<COLOR_FLOAT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((float*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<COLOR_HALF>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((__half*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<COLOR_8BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint8_t*)(source_plane+line*stride))[column]) / 255.f;
}

template<>
__device__ float inline PickValue<COLOR_9BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 511.f;
}

template<>
__device__ float inline PickValue<COLOR_10BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 1023.f;
}

template<>
__device__ float inline PickValue<COLOR_12BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 4095.f;
}

template<>
__device__ float inline PickValue<COLOR_14BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 16383.f;
}

template<>
__device__ float inline PickValue<COLOR_16BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return (float)(((uint16_t*)(source_plane+line*stride))[column]) / 65535.f;
}

template<Sample_Type T>
__global__ void convertToFloatPlane_Kernel(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width*height) return;

    output_plane[x] = PickValue<T>(source_plane, x, stride, width);
}

template<Sample_Type T>
__host__ void inline convertToFloatPlane(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, hipStream_t stream){
    const int thx = 256;
    const int blx = (width*height + thx -1)/thx;
    convertToFloatPlane_Kernel<T><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
}

__host__ bool inline convertToFloatPlaneSwitch(float* output_plane, const uint8_t* const source_plane, const int stride, const int width, const int height, Sample_Type T, hipStream_t stream){
    const int thx = 256;
    const int blx = (width*height + thx -1)/thx;
    switch (T){
        case COLOR_FLOAT:
            convertToFloatPlane_Kernel<COLOR_FLOAT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_HALF:
            convertToFloatPlane_Kernel<COLOR_HALF><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_8BIT:
            convertToFloatPlane_Kernel<COLOR_8BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_9BIT:
            convertToFloatPlane_Kernel<COLOR_9BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_10BIT:
            convertToFloatPlane_Kernel<COLOR_10BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_12BIT:
            convertToFloatPlane_Kernel<COLOR_12BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_14BIT:
            convertToFloatPlane_Kernel<COLOR_14BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        case COLOR_16BIT:
            convertToFloatPlane_Kernel<COLOR_16BIT><<<dim3(blx), dim3(thx), 0, stream>>>(output_plane, source_plane, stride, width, height);
        break;
        default:
            return 1;
    }
    return 0;
}

}