#pragma once

#include <string>

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"
#include "../util/Planed.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"

#include "gaussianblur.hpp" 
#include "downupsample.hpp"
#include "colors.hpp" //OpsinDynamicsImage
#include "separatefrequencies.hpp"
#include "maltaDiff.hpp"
#include "simplerdiff.hpp" //L2 +asym diff + same noise diff
#include "maskPsycho.hpp"
#include "combineMasks.hpp"
#include "diffnorms.hpp" //takes diffmap and returns norm2, norm3 and norminf

namespace butter{

float* getdiffmap(float* src1_d[3], float* src2_d[3], float* mem_d, int64_t width, int64_t height, float intensity_multiplier, int64_t maxshared, GaussianHandle& gaussianHandle, hipStream_t stream){
    //Psycho Image planes

    float* temp[3] = {mem_d, mem_d+1*width*height, mem_d+2*width*height};

    float* lf1[3] = {mem_d+3*width*height, mem_d+4*width*height, mem_d+5*width*height};
    float* mf1[3] = {mem_d+6*width*height, mem_d+7*width*height, mem_d+8*width*height};
    float* hf1[3] = {mem_d+9*width*height, mem_d+10*width*height};
    float* uhf1[3] = {mem_d+11*width*height, mem_d+12*width*height};

    float* lf2[3] = {mem_d+13*width*height, mem_d+14*width*height, mem_d+15*width*height};
    float* mf2[3] = {mem_d+16*width*height, mem_d+17*width*height, mem_d+18*width*height};
    float* hf2[3] = {mem_d+19*width*height, mem_d+20*width*height};
    float* uhf2[3] = {mem_d+21*width*height, mem_d+22*width*height};

    //to XYB
    opsinDynamicsImage(src1_d, temp, width, height, gaussianHandle, intensity_multiplier, stream);
    opsinDynamicsImage(src2_d, temp, width, height, gaussianHandle, intensity_multiplier, stream);

    separateFrequencies(src1_d, temp, lf1, mf1, hf1, uhf1, width, height, gaussianHandle, stream);
    separateFrequencies(src2_d, temp, lf2, mf2, hf2, uhf2, width, height, gaussianHandle, stream);

    //no more needs for src1_d and src2_d so we reuse them as masks for butter
    float* block_diff_dc[3] = {src1_d[0], src1_d[1], src1_d[2]};
    float* block_diff_ac[3] = {src2_d[0], src2_d[1], src2_d[2]};

    //set the accumulators to 0
    for (int c = 0; c < 3; c++){
        hipMemsetAsync(block_diff_ac[c], 0, sizeof(float)*width*height, stream);
        hipMemsetAsync(block_diff_dc[c], 0, sizeof(float)*width*height, stream);
    }

    const float hf_asymmetry_ = 0.8f;

    const float wUhfMalta = 1.10039032555f;
    const float norm1Uhf = 71.7800275169f;
    MaltaDiffMap(uhf1[1], uhf2[1], block_diff_ac[1], width, height, wUhfMalta * hf_asymmetry_, wUhfMalta / hf_asymmetry_, norm1Uhf, stream);

    const float wUhfMaltaX = 173.5f;
    const float norm1UhfX = 5.0f;
    MaltaDiffMap(uhf1[0], uhf2[0], block_diff_ac[0], width, height, wUhfMaltaX * hf_asymmetry_, wUhfMaltaX / hf_asymmetry_, norm1UhfX, stream);

    const float wHfMalta = 18.7237414387f;
    const float norm1Hf = 4498534.45232f;
    MaltaDiffMapLF(hf1[1], hf2[1], block_diff_ac[1], width, height, wHfMalta * std::sqrt(hf_asymmetry_), wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf, stream);

    const float wHfMaltaX = 6923.99476109f;
    const float norm1HfX = 8051.15833247f;
    MaltaDiffMapLF(hf1[0], hf2[0], block_diff_ac[0], width, height, wHfMaltaX * std::sqrt(hf_asymmetry_), wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX, stream);

    const float wMfMalta = 37.0819870399f;
    const float norm1Mf = 130262059.556f;
    MaltaDiffMapLF(mf1[1], mf2[1], block_diff_ac[1], width, height, wMfMalta, wMfMalta, norm1Mf, stream);

    const float wMfMaltaX = 8246.75321353f;
    const float norm1MfX = 1009002.70582f;
    MaltaDiffMapLF(mf1[0], mf2[0], block_diff_ac[0], width, height, wMfMaltaX, wMfMaltaX, norm1MfX, stream);

    const float wmul[9] = {
      400.0f,         1.50815703118f,  0.0f,
      2150.0f,        10.6195433239f,  16.2176043152f,
      29.2353797994f, 0.844626970982f, 0.703646627719f,
    };

    for (int c = 0; c < 3; c++){
        if (c < 2){
            L2AsymDiff(hf1[c], hf2[c], block_diff_ac[c], width*height, wmul[c] * hf_asymmetry_, wmul[c] / hf_asymmetry_, stream);
        }
        L2diff(mf1[c], mf2[c], block_diff_ac[c], width*height, wmul[3 + c], stream);
        L2diff(lf1[c], lf2[c], block_diff_dc[c], width*height, wmul[6 + c], stream);
    }

    //from now on, lf and mf are not used so we will reuse the memory
    float* mask = temp[1];
    float* temp3[3] = {lf2[0], lf2[1], lf2[2]};
    float* temp4[3] = {mf2[0], mf2[1], mf2[2]};

    MaskPsychoImage(hf1, uhf1, hf2, uhf2, temp3[0], temp4[0], mask, block_diff_ac, width, height, gaussianHandle, stream);
    //at this point hf and uhf cannot be used anymore (they have been invalidated by the function)

    float* diffmap_ = temp[0];
    computeDiffmap(mask, block_diff_dc[0], block_diff_dc[1], block_diff_dc[2], block_diff_ac[0], block_diff_ac[1], block_diff_ac[2], diffmap_, width*height, stream);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!
    //GPU_CHECK(hipGetLastError());

    //printf("End result: %f, %f and %f\n", norm2, norm3, norminf);
    
    return diffmap_;
}

//expects linear planar RGB as input, mem_d must contain 25 planes (srcs are rewritten so they are unusable after execution)
float* getmultiscalediffmap(float* src1_d[3], float* src2_d[3], float* mem_d, int64_t width, int64_t height, float intensity_multiplier, int64_t maxshared, GaussianHandle& gaussianHandle, hipStream_t stream){
    //computing downscaled before we overwrite src in getdiffmap (it s better for memory)
    int64_t nwidth = (width-1)/2+1;
    int64_t nheight = (height-1)/2+1;
    float* nmem_d = mem_d; //allow usage up to mem_d+2*width*height;
    float* nsrc1_d[3] = {nmem_d, nmem_d+nwidth*nheight, nmem_d+2*nwidth*nheight};
    float* nsrc2_d[3] = {nmem_d+3*nwidth*nheight, nmem_d+4*nwidth*nheight, nmem_d+5*nwidth*nheight};

    //using 6 smaller planes is equivalent to 1.5 standard planes, so it fits within the 2 planes given here!)
    for (int i = 0; i < 3; i++){
        downsample(src1_d[i], nsrc1_d[i], width, height, stream);
        downsample(src2_d[i], nsrc2_d[i], width, height, stream);
    }

    float* diffmap = getdiffmap(src1_d, src2_d, mem_d+2*width*height, width, height, intensity_multiplier, maxshared, gaussianHandle, stream);
    //diffmap is stored at mem_d+8*width*height so we can build after that the second smaller scale
    //smaller scale now
    nmem_d = mem_d+3*width*height;
    float* diffmapsmall = getdiffmap(nsrc1_d, nsrc2_d, nmem_d+6*nwidth*nheight, nwidth, nheight, intensity_multiplier, maxshared, gaussianHandle, stream);

    addsupersample2X(diffmap, diffmapsmall, width, height, 0.5f, stream);

    return diffmap;
}

std::tuple<float, float, float> butterprocess(const uint8_t *dstp, int64_t dststride, const uint8_t *srcp1[3], const uint8_t *srcp2[3], float* pinned, GaussianHandle& gaussianHandle, VshipColorConvert::Converter converter1, VshipColorConvert::Converter converter2, const int64_t lineSize[3], const int64_t lineSize2[3], int64_t width, int64_t height, int Qnorm, float intensity_multiplier, int64_t maxshared, hipStream_t stream){
    int64_t wh = width*height;
    const int64_t totalscalesize = wh;

    hipError_t erralloc;

    const int totalplane = 31;
    float* mem_d;
    erralloc = hipMallocAsync(&mem_d, sizeof(float)*totalscalesize*(totalplane), stream); //max just in case stride is ridiculously large
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }
    //initial color planes
    float* src1_d[3] = {mem_d, mem_d+width*height, mem_d+2*width*height};
    float* src2_d[3] = {mem_d+3*width*height, mem_d+4*width*height, mem_d+5*width*height};
    
    converter1.convert(src1_d, srcp1, lineSize);
    converter2.convert(src2_d, srcp2, lineSize2);

    float* diffmap = getmultiscalediffmap(src1_d, src2_d, mem_d+6*width*height, width, height, intensity_multiplier, maxshared, gaussianHandle, stream);

    //diffmap is in its final form
    if (dstp != NULL){
        strideAdder(diffmap, mem_d+6*width*height, dststride, width, height, stream);
        GPU_CHECK(hipMemcpyDtoHAsync((void*)(dstp), mem_d+6*width*height, dststride * height, stream));
    }

    std::tuple<float, float, float> finalres;
    try{
        finalres = diffmapscore(diffmap, mem_d+9*width*height, mem_d+10*width*height, pinned, width*height, Qnorm, stream);
    } catch (const VshipError& e){
        hipFree(mem_d);
        throw e;
    }

    hipFreeAsync(mem_d, stream);

    return finalres;
}

class ButterComputingImplementation{
    float* pinned;
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    int Qnorm;
    float intensity_multiplier;
    int64_t width;
    int64_t height;
    int maxshared;
    hipStream_t stream;
public:
    //Qnorm replace the old norm2 and allows getting really any norm wanted
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int Qnorm, float intensity_multiplier){
        hipStreamCreate(&stream);
        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, stream);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, stream);

        this->width = converter1.getWidth();
        this->height = converter1.getHeight();

        //assert they have the same width/height
        if (converter2.getWidth() != width || converter2.getHeight() != height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        this->Qnorm = Qnorm;
        this->intensity_multiplier = intensity_multiplier;

        gaussianhandle.init();

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;

        const int64_t pinnedsize = allocsizeScore(width, height);
        hipError_t erralloc = hipHostMalloc(&pinned, sizeof(float)*pinnedsize);
        if (erralloc != hipSuccess){
            gaussianhandle.destroy();
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }
    }
    void destroy(){
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        hipStreamDestroy(stream);
        hipHostFree(pinned);
    }
    //if dstp is NULL, distmap won't be retrieved
    std::tuple<float, float, float> run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        return butterprocess(dstp, dststride, srcp1, srcp2, pinned, gaussianhandle, converter1, converter2, lineSize, lineSize2, width, height, Qnorm, intensity_multiplier, maxshared, stream);
    }
};

}
