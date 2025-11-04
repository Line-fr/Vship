#pragma once

//define enums for colorspaces
#include "../VshipColor.h"

#include "anyDepthToFloat.hpp"
#include "primariesToBT709.hpp"
#include "transferToLinear.hpp"
#include "chromaUpsample.hpp"

namespace VshipColorConvert{

//accept only YUV format
int extractInfoFromPixelFormat(AVPixelFormat pix_fmt, Vship_Sample_t& sample_type, int& subw, int& subh){
    switch (pix_fmt){
        case AV_PIX_FMT_YUV420P:
            sample_type = Vship_SampleUINT8;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P:
            sample_type = Vship_SampleUINT8;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P:
            sample_type = Vship_SampleUINT8;
            subw = 0;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV410P:
            sample_type = Vship_SampleUINT8;
            subw = 2;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV411P:
            sample_type = Vship_SampleUINT8;
            subw = 2;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV440P:
            sample_type = Vship_SampleUINT8;
            subw = 0;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV420P16:
            sample_type = Vship_SampleUINT16;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P16:
            sample_type = Vship_SampleUINT16;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P16:
            sample_type = Vship_SampleUINT16;
            subw = 0;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV420P9:
            sample_type = Vship_SampleUINT9;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P9:
            sample_type = Vship_SampleUINT9;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P9:
            sample_type = Vship_SampleUINT9;
            subw = 0;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV420P10:
            sample_type = Vship_SampleUINT10;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P10:
            sample_type = Vship_SampleUINT10;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P10:
            sample_type = Vship_SampleUINT10;
            subw = 0;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV440P10:
            sample_type = Vship_SampleUINT10;
            subw = 0;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV420P12:
            sample_type = Vship_SampleUINT10;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P12:
            sample_type = Vship_SampleUINT12;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P12:
            sample_type = Vship_SampleUINT12;
            subw = 0;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV420P14:
            sample_type = Vship_SampleUINT14;
            subw = 1;
            subh = 1;
            break;
        case AV_PIX_FMT_YUV422P14:
            sample_type = Vship_SampleUINT14;
            subw = 1;
            subh = 0;
            break;
        case AV_PIX_FMT_YUV444P14:
            sample_type = Vship_SampleUINT14;
            subw = 0;
            subh = 0;
            break;
        default:
            return 1;
    }
    return 0;
}

//strides are for the source, width and height are for end video output (generally luma plane)
int linearize(float* outplane[3], float* tempplane[3], const uint8_t* source_plane[3], int strides[3], int width, int height, AVPixelFormat pix_fmt, AVChromaLocation location, hipStream_t stream){
    //Vship_Colorspace_t colorspace;

    /*
    if (extractInfoFromPixelFormat(pix_fmt, sample_type, subw, subh) != 0) {
        std::cout << "pixel format not supported : " << (int)pix_fmt << std::endl;
        return 1;
    }

    //first step, transform current integer/float format into a pure float
    bool res = 1;
    res &= convertToFloatPlane(outplane[0], source_plane[0], strides[0], width, height, sample_type, stream); //get to outplane directly
    res &= convertToFloatPlane(tempplane[1], source_plane[1], strides[1], width >> subw, height >> subh, sample_type, stream);
    res &= convertToFloatPlane(tempplane[2], source_plane[2], strides[2], width >> subw, height >> subh, sample_type, stream);
    if (res != 0) {
        std::cout << "sample_type not supported" << std::endl;
        return res;
    }

    //second step, chroma upsample
    if (upsample(outplane, tempplane, width, height, location, subw, subh, stream) != 0){ //this function does not transfer luma plane from temp to out!! so we directly get luma plane to out instead of temp since no modification is needed
        std::cout << "Failed to upscale" << std::endl;
        return 1;
    }
    */

    return 0;
}

}