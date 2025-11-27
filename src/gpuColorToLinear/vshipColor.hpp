#pragma once

#include "../util/float3operations.hpp"

//define enums for colorspaces
#include "../VshipColor.h"

#include "resize.hpp"
#include "anyDepthToFloat.hpp" //manages range and sample type
#include "chromaUpsample.hpp" //manages chroma location and chroma subsampling
#include "YUVToLinRGB.hpp" //manages transfer function and YUV Matrix
#include "primaries.hpp" //manages primaries
#include "../util/Planed.hpp"

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

    bool isCropped;
    bool isResized;
public:
    Converter() = default;
    //width and height are of the input (before crop)
    void init(Vship_Colorspace_t colorspace, ConverterDestination_t destination, hipStream_t stream){
        this->width = colorspace.width;
        this->height = colorspace.height;
        this->source_colorspace = colorspace;
        this->destination = destination;
        this->stream = stream;       

        isCropped = true;
        if (colorspace.crop.top == 0 && colorspace.crop.bottom == 0 && colorspace.crop.left == 0 && colorspace.crop.right == 0) isCropped = false;

        if (colorspace.target_width == -1) source_colorspace.target_width = colorspace.width;
        if (colorspace.target_height == -1) source_colorspace.target_height = colorspace.height;
        isResized = true;
        if (source_colorspace.target_width == colorspace.width && source_colorspace.target_height == colorspace.height) isResized = false;
        int64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        int64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
        
    }
    void destroy(){
    }
    int64_t getWidth(){
        return source_colorspace.target_width - source_colorspace.crop.left - source_colorspace.crop.right;
    }
    int64_t getHeight(){
        return source_colorspace.target_height - source_colorspace.crop.top - source_colorspace.crop.bottom;
    }
    void convert(float* out[3], const uint8_t *inp[3], const int64_t lineSize[3]){
        int64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        int64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);

        int basePlaneAllocation = 3;
        int maxplaneBuffer = 2;
        if (!isResized && !isCropped) basePlaneAllocation = 0;
        hipError_t erralloc = hipMallocAsync(&mem_d, sizeof(float)*width*height*basePlaneAllocation + sizeof(float)*maxWidth*maxHeight*maxplaneBuffer, stream);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }

        float* preCropOut[3] = {mem_d, mem_d+width*height, mem_d+width*height*2};
        float* temp_d = (mem_d+3*width*height);
        if (!isResized && !isCropped) {
            //direct feed
            for (int i = 0; i < 3; i++) preCropOut[i] = out[i];
            temp_d = mem_d;
        }

        uint8_t* src_d = (uint8_t*)temp_d;
        int64_t maxstride = std::max(lineSize[0], std::max(lineSize[1], lineSize[2]));
        if (maxstride*height > sizeof(float)*maxWidth*maxHeight*2){
            //we need to allocate another plane to export the current data to gpu
            hipError_t erralloc = hipMallocAsync(&src_d, maxstride*height, stream);
            if (erralloc != hipSuccess){
                throw VshipError(OutOfVRAM, __FILE__, __LINE__);
            }
        }
        //before chroma upsampling
        const int64_t plane_widths[3] = {width, width >> source_colorspace.subsampling.subw, width >> source_colorspace.subsampling.subw};
        const int64_t plane_heights[3] = {height, height >> source_colorspace.subsampling.subh, height >> source_colorspace.subsampling.subh};
        const int byteSize = bytesizeSample(source_colorspace.sample);
        for (int i = 0; i < 3; i++){
            hipMemcpyHtoDAsync(src_d, (void*)(inp[i]), lineSize[i]*plane_heights[i] - (lineSize[i] - byteSize*plane_widths[i]), stream);
            convertToFloatPlane(preCropOut[i], (uint8_t*)src_d, lineSize[i], plane_widths[i], plane_heights[i], source_colorspace.sample, source_colorspace.range, source_colorspace.colorFamily, (bool)(i != 0), stream);
        }
        if (maxstride*height > sizeof(float)*maxWidth*maxHeight*2){
            hipFreeAsync(src_d, stream);
        }
        //now we have our float data with the right range in out
        //let's upsample chroma using mem_d as a temporary plane
        upsample(temp_d, preCropOut, width, height, source_colorspace.chromaLocation, source_colorspace.subsampling, stream);

        //now, we need to transform the YUV into linRGB
        YUVToLinRGBPipeline(preCropOut[0], preCropOut[1], preCropOut[2], width*height, source_colorspace.YUVMatrix, source_colorspace.transferFunction, stream);


        //then we manage primaries, it depends on the destination
        switch(destination){
            case linRGB:
                break; //we are done already
            case XYZ:
                if (source_colorspace.primaries != Vship_PRIMARIES_INTERNAL){
                    primariesToPrimaries(preCropOut[0], preCropOut[1], preCropOut[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_INTERNAL, stream);
                }
                break;
            case linRGBBT709:
                if (source_colorspace.primaries != Vship_PRIMARIES_BT709){
                    primariesToPrimaries(preCropOut[0], preCropOut[1], preCropOut[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_BT709, stream);
                }
                break;
        }

        //now we crop and put in out
        for (int i = 0; i < 3; i++){
            //resize and crop (we are still in linear space)
            float* base = preCropOut[i];
            float* interm = temp_d;
            float* interm2 = temp_d + maxWidth*maxHeight;
            float* fin = out[i];
            if (isCropped){
                //std::cout << source_colorspace.width << "x" << source_colorspace.height << " / " << source_colorspace.target_width << "x" << source_colorspace.target_height << std::endl;
                //std::cout << source_colorspace.crop.top << " " << source_colorspace.crop.bottom << " " << source_colorspace.crop.left << " " << source_colorspace.crop.right << std::endl;
                resizePlane(interm, interm2, base, source_colorspace.width, source_colorspace.height, source_colorspace.target_width, source_colorspace.target_height, stream);
                strideEliminator<FLOAT>(fin, interm+source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left, source_colorspace.target_width*sizeof(float), getWidth(), getHeight(), stream);
            } else if (isCropped){
                //std::cout << "Pointer offset: " << source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left << ", stride: " << source_colorspace.target_width*sizeof(float) << ", new width: " << getWidth() << ", new height: " << getHeight() << std::endl;
                strideEliminator<FLOAT>(fin, base+source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left, source_colorspace.target_width*sizeof(float), getWidth(), getHeight(), stream);
            } else if (isResized){
                resizePlane(fin, interm, base, source_colorspace.width, source_colorspace.height, source_colorspace.target_width, source_colorspace.target_height, stream);
            }
        }
        hipFreeAsync(mem_d, stream);
    }
};

}