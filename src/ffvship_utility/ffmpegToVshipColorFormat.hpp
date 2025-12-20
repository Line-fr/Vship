#pragma once

int ffmpegToVshipFormat(Vship_Colorspace_t& out, const FFMS_Frame* in){
    out.width = in->EncodedWidth;
    out.height = in->EncodedHeight;

    //default values
    out.sample = Vship_SampleUINT8;
    out.colorFamily = Vship_ColorYUV;
    out.YUVMatrix = (out.height > 650) ? Vship_MATRIX_BT709 : Vship_MATRIX_BT470_BG;
    out.transferFunction = Vship_TRC_BT709;
    out.primaries = Vship_PRIMARIES_BT709;
    out.range = Vship_RangeLimited;
    switch ((AVPixelFormat)in->EncodedPixelFormat){
        case AV_PIX_FMT_YUVA420P:
        case AV_PIX_FMT_YUV420P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YUYV422:
        case AV_PIX_FMT_YUVA422P:
        case AV_PIX_FMT_YUV422P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA444P:
        case AV_PIX_FMT_YUV444P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV410P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 2;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUV411P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 2;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV440P:
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 0;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVA420P16LE:
        case AV_PIX_FMT_YUV420P16LE:
            out.sample = Vship_SampleUINT16;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVA422P16LE:
        case AV_PIX_FMT_YUV422P16LE:
            out.sample = Vship_SampleUINT16;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA444P16LE:
        case AV_PIX_FMT_YUV444P16LE:
            out.sample = Vship_SampleUINT16;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA420P9LE:
        case AV_PIX_FMT_YUV420P9LE:
            out.sample = Vship_SampleUINT9;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVA420P10LE:
        case AV_PIX_FMT_YUV420P10LE:
            out.sample = Vship_SampleUINT10;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVA422P10LE:
        case AV_PIX_FMT_YUV422P10LE:
            out.sample = Vship_SampleUINT10;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA444P9LE:
        case AV_PIX_FMT_YUV444P9LE:
            out.sample = Vship_SampleUINT9;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA444P10LE:
        case AV_PIX_FMT_YUV444P10LE:
            out.sample = Vship_SampleUINT10;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA422P9LE:
        case AV_PIX_FMT_YUV422P9LE:
            out.sample = Vship_SampleUINT9;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV420P12LE:
            out.sample = Vship_SampleUINT12;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUV420P14LE:
            out.sample = Vship_SampleUINT14;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVA422P12LE:
        case AV_PIX_FMT_YUV422P12LE:
            out.sample = Vship_SampleUINT12;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV422P14LE:
            out.sample = Vship_SampleUINT14;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVA444P12LE:
        case AV_PIX_FMT_YUV444P12LE:
            out.sample = Vship_SampleUINT12;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV444P14LE:
            out.sample = Vship_SampleUINT14;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUV440P10LE:
            out.sample = Vship_SampleUINT10;
            out.subsampling.subw = 0;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUV440P12LE:
            out.sample = Vship_SampleUINT12;
            out.subsampling.subw = 0;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVJ420P:
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 1;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_YUVJ422P:
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 1;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVJ444P:
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVJ411P:
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 2;
            out.subsampling.subh = 0;
        break;
        case AV_PIX_FMT_YUVJ440P:
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 0;
            out.subsampling.subh = 1;
        break;
        case AV_PIX_FMT_RGB48LE:
        case AV_PIX_FMT_RGBA64LE:
            out.sample = Vship_SampleUINT16;
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_BGRA:
            out.colorFamily = Vship_ColorRGB;
            out.YUVMatrix = Vship_MATRIX_RGB;
            out.transferFunction = Vship_TRC_sRGB;
            out.primaries = Vship_PRIMARIES_BT709;
            out.range = Vship_RangeFull;
            out.sample = Vship_SampleUINT8;
            out.subsampling.subw = 0;
            out.subsampling.subh = 0;
        break;
        default:
            std::cerr << "Unhandled LibAV Pixel Format " << (AVPixelFormat)in->EncodedPixelFormat << std::endl;
            return 1;
    }

    /*if (out.width%(1 << out.subsampling.subw) != 0 || out.height%(1 << out.subsampling.subh)){
        std::cerr << "Width and height are not compatible with the subsampling. (For example odd width in YUV4:2:0). This is not supported by vship" << std::endl;
        return 1;
    }*/

    switch ((FFMS_ChromaLocations)in->ChromaLocation){
        case FFMS_LOC_UNSPECIFIED:
            //std::cerr << "unspecifed chroma location, defaulting on left" << std::endl;
        case FFMS_LOC_LEFT:
            out.chromaLocation = Vship_ChromaLoc_Left;
            break;
        case FFMS_LOC_CENTER:
            out.chromaLocation = Vship_ChromaLoc_Center;
            break;
        case FFMS_LOC_TOPLEFT:
            out.chromaLocation = Vship_ChromaLoc_TopLeft;
            break;
        case FFMS_LOC_TOP:
            out.chromaLocation = Vship_ChromaLoc_Top;
            break;
        /*
        case FFMS_LOC_BOTTOMLEFT:
            out.chroma_location = ZIMG_CHROMA_BOTTOM_LEFT;
            break;
        case FFMS_LOC_BOTTOM:
            out.chroma_location = ZIMG_CHROMA_BOTTOM;
            break;
        */

        default:
            std::cerr << "Unhandled LibAV Chroma position: " << in->ChromaLocation << std::endl;
            return 1;
    }
    
    switch ((AVColorSpace)in->ColorSpace){
        case AVCOL_SPC_RGB:
            out.YUVMatrix = Vship_MATRIX_RGB;
            break;
        case AVCOL_SPC_BT709:    
            out.YUVMatrix = Vship_MATRIX_BT709;                  
            break;
        case AVCOL_SPC_UNSPECIFIED:
            //std::cerr << "missing YUV matrix color, guessing..." << std::endl;                 
            break;
        /*
        case AVCOL_SPC_FCC:    
            out.matrix_coefficients = ZIMG_MATRIX_FCC;            
            break;
        */
        case AVCOL_SPC_BT470BG:    
            out.YUVMatrix = Vship_MATRIX_BT470_BG;           
            break;
        case AVCOL_SPC_SMPTE170M:    
            out.YUVMatrix = Vship_MATRIX_ST170_M;                 
            break;
        /*
        case AVCOL_SPC_SMPTE240M:    
            out.matrix_coefficients = ZIMG_MATRIX_ST240_M;                 
            break;
        case AVCOL_SPC_YCGCO:    
            out.matrix_coefficients = ZIMG_MATRIX_YCGCO;                   
            break;
        */
        case AVCOL_SPC_BT2020_NCL:    
            out.YUVMatrix = Vship_MATRIX_BT2020_NCL;
            break;
        case AVCOL_SPC_BT2020_CL:    
            out.YUVMatrix = Vship_MATRIX_BT2020_CL;             
            break;
        /*
        //case AVCOL_SPC_SMPTE2085:    
        //    out.matrix_coefficients = ZIMG_MATRIX_ST2085_YDZDX;            
        //    break;
        case AVCOL_SPC_CHROMA_DERIVED_NCL:    
            out.matrix_coefficients = ZIMG_MATRIX_CHROMATICITY_DERIVED_NCL;
            break;
        case AVCOL_SPC_CHROMA_DERIVED_CL:    
            out.matrix_coefficients = ZIMG_MATRIX_CHROMATICITY_DERIVED_CL; 
            break;
        //case AVCOL_SPC_ICTCP:    
        //    out.matrix_coefficients = ZIMG_MATRIX_BT2100_ICTCP;            
        //    break;
        */
        default:
            std::cerr << "Unhandled LibAV YUV color matrix: " << in->ColorSpace << std::endl;
            return 1;
    }

    if (out.YUVMatrix == Vship_MATRIX_BT470_BG){
        //new default
        out.transferFunction = Vship_TRC_BT470_BG;
        out.primaries = Vship_PRIMARIES_BT470_BG;   
    } else if (out.YUVMatrix == Vship_MATRIX_BT2020_CL || out.YUVMatrix == Vship_MATRIX_BT2020_NCL || out.YUVMatrix == Vship_MATRIX_BT2100_ICTCP){
        //default to WCG and HDR.
        out.transferFunction = Vship_TRC_PQ;
        out.primaries = Vship_PRIMARIES_BT2020;
    }
    
    switch (in->TransferCharateristics){
        case AVCOL_TRC_UNSPECIFIED:
            //std::cerr << "missing transfer function, using BT709" << std::endl;;
            break;    
        case AVCOL_TRC_BT709:
            out.transferFunction = Vship_TRC_BT709;
            break;
        case AVCOL_TRC_GAMMA22:
            out.transferFunction = Vship_TRC_BT470_M;
            break;
        case AVCOL_TRC_GAMMA28:
            out.transferFunction = Vship_TRC_BT470_BG;
            break;
        case AVCOL_TRC_SMPTE170M:
            out.transferFunction = Vship_TRC_BT601;
            break;
        case AVCOL_TRC_SMPTE240M:
            out.transfer_characteristics = Vship_TRC_ST240_M;
            break;
        case AVCOL_TRC_LINEAR:
            out.transferFunction = Vship_TRC_Linear;
            break;
        /*
        case AVCOL_TRC_LOG:
            out.transfer_characteristics = ZIMG_TRANSFER_LOG_100;
            break;
        case AVCOL_TRC_LOG_SQRT:
            out.transfer_characteristics = ZIMG_TRANSFER_LOG_316;
            break;
        case AVCOL_TRC_IEC61966_2_4:
            out.transfer_characteristics = ZIMG_TRANSFER_IEC_61966_2_4;
            break;
        //case AVCOL_TRC_BT1361_ECG:
        //    out.transfer_characteristics = ZIMG_TRANSFER_BT1361;
        //    break;
        */
        case AVCOL_TRC_IEC61966_2_1:
            out.transferFunction = Vship_TRC_sRGB;
            break;
        case AVCOL_TRC_BT2020_10:
            out.transferFunction = Vship_TRC_BT709;
            break;
        case AVCOL_TRC_BT2020_12:
            out.transferFunction = Vship_TRC_BT709;
            break;
        case AVCOL_TRC_SMPTE2084:
            out.transferFunction = Vship_TRC_PQ;
            break;
        case AVCOL_TRC_SMPTE428:
            out.transferFunction = Vship_TRC_ST428;
            break;
        case AVCOL_TRC_ARIB_STD_B67:
            out.transferFunction = Vship_TRC_HLG;
            break;

        default:
            std::cerr << "Unhandled LibAV color transfer function: " << in->TransferCharateristics << std::endl;
            return 1;
    }
    
    switch (in->ColorPrimaries){
        case AVCOL_PRI_UNSPECIFIED:
            //std::cerr << "unspecified primaries, defaulting to BT709" << std::endl;
            break;
        case AVCOL_PRI_BT709:
            out.primaries = Vship_PRIMARIES_BT709;
            break;
        case AVCOL_PRI_BT470M:
            out.primaries = Vship_PRIMARIES_BT470_M;
            break;
        case AVCOL_PRI_BT470BG:
            out.primaries = Vship_PRIMARIES_BT470_BG;
            break;
        case AVCOL_PRI_SMPTE170M:
            out.primaries = Vship_PRIMARIES_ST170_M;
            break;
        case AVCOL_PRI_SMPTE240M:
            out.primaries = Vship_PRIMARIES_ST240_M;
            break;
        /*
        case AVCOL_PRI_SMPTE170M:
            out.color_primaries = ZIMG_PRIMARIES_ST170_M;
            break;
        case AVCOL_PRI_SMPTE240M:
            out.color_primaries = ZIMG_PRIMARIES_ST240_M;
            break;
        case AVCOL_PRI_FILM:
            out.color_primaries = ZIMG_PRIMARIES_FILM;
            break;
        */
        case AVCOL_PRI_BT2020:
            out.primaries = Vship_PRIMARIES_BT2020;
            break;
        /*
        case AVCOL_PRI_SMPTE428:
            out.color_primaries = ZIMG_PRIMARIES_ST428;
            break;
        case AVCOL_PRI_SMPTE431:
            out.color_primaries = ZIMG_PRIMARIES_ST431_2;
            break;
        case AVCOL_PRI_SMPTE432:
            out.color_primaries = ZIMG_PRIMARIES_ST432_1;
            break;
        case AVCOL_PRI_EBU3213:
            out.color_primaries = ZIMG_PRIMARIES_EBU3213_E;
            break;
        */
            
        default:
            std::cerr << "Unhandled LibAV color primaries: " << in->ColorPrimaries << std::endl;
            return 1;
    }

    switch (in->ColorRange){
        case AVCOL_RANGE_UNSPECIFIED:
            //std::cerr << "Warning: unspecified color range, defaulting to full" << std::endl;
            break;
        case AVCOL_RANGE_MPEG:
            out.range = Vship_RangeLimited;
            break;
        case AVCOL_RANGE_JPEG:
            out.range = Vship_RangeFull;
            break;
        default:
            std::cerr << "Unhandled LibAV color range object received " << std::endl;
            return 1;
    }
    return 0;
}