#pragma once

//define enums for colorspaces
#include "../VshipColor.h"

#include "anyDepthToFloat.hpp" //manages range and sample type
#include "chromaUpsample.hpp" //manages chroma location and chroma subsampling
#include "YUVToLinRGB.hpp" //manages transfer function and YUV Matrix
#include "primaries.hpp" //manages primaries

namespace VshipColorConvert{

enum ConverterDestination_t{
    XYZ,
    linRGBBT709,
    linRGB, //no primaries conversion
};

class Converter{
    Vship_Colorspace_t source_colorspace;
    ConverterDestination_t destination;
    int64_t width;
    int64_t height;
    hipStream_t stream;
    float* mem_d = NULL;
public:
    Converter() = default;
    void init(int64_t width, int64_t height, Vship_Colorspace_t colorspace, ConverterDestination_t destination, hipStream_t stream){
        this->width = width;
        this->height = height;
        this->source_colorspace = colorspace;
        this->destination = destination;
        this->stream = stream;

        //initiliaze buffer -> 2 frames for chroma upsampling if needed
        if (colorspace.subsampling.subw != 0 || colorspace.subsampling.subh != 0){
            hipError_t erralloc = hipMalloc(&mem_d, sizeof(float)*width*height*2);
            if (erralloc != hipSuccess){
                throw VshipError(OutOfVRAM, __FILE__, __LINE__);
            }
        }
    }
    void destroy(){
        if (mem_d != NULL){
            hipFree(mem_d);
            mem_d = NULL;
        }
    }
    void convert(float* out[3], const uint8_t *inp[3], int64_t stride){
        uint8_t* src_d = (uint8_t*)mem_d;
        if (stride > sizeof(float)*width*2){
            //we need to allocate another plane to export the current data to gpu
            hipError_t erralloc = hipMallocAsync(&src_d, stride*height, stream);
            if (erralloc != hipSuccess){
                throw VshipError(OutOfVRAM, __FILE__, __LINE__);
            }
        }
        for (int i = 0; i < 3; i++){
            hipMemcpyHtoDAsync(src_d, inp[i], stride*height, stream);
            convertToFloatPlane(out[i], (uint8_t*)src_d, stride, width, height, source_colorspace.sample, source_colorspace.range, stream);
        }
        if (stride > sizeof(float)*width*2){
            hipFreeAsync(src_d, stream);
        }
        //now we have our float data with the right range in out
        //let's upsample chroma using mem_d as a temporary plane
        upsample(mem_d, out, width, height, source_colorspace.chromaLocation, source_colorspace.subsampling, stream);

        //now, we need to transform the YUV into linRGB
        YUVToLinRGBPipeline(out[0], out[1], out[2], width*height, source_colorspace.YUVMatrix, source_colorspace.transferFunction, stream);

        //then we manage primaries, it depends on the destination
        switch(destination){
            case linRGB:
                break; //we are done already
            case XYZ:
                if (source_colorspace.primaries != Vship_PRIMARIES_INTERNAL){
                    primariesToPrimaries(out[0], out[1], out[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_INTERNAL, stream);
                }
                break;
            case linRGBBT709:
                if (source_colorspace.primaries != Vship_PRIMARIES_BT709){
                    primariesToPrimaries(out[0], out[1], out[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_BT709, stream);
                }
                break;
        }
    }
};

}