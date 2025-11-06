#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"
#include "makeXYB.hpp"
#include "downsample.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

namespace ssimu2{

__launch_bounds__(256)
__global__ void memoryorganizer_kernel(float3* out, float* srcp0, float* srcp1, float* srcp2, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    out[i*width + j].x = srcp0[i*width+j];
    out[i*width + j].y = srcp1[i*width+j];
    out[i*width + j].z = srcp2[i*width+j];
}

void memoryorganizer(float3* out, float* srcp0, float* srcp1, float* srcp2, int64_t width, int64_t height, hipStream_t stream){
    int th_x = std::min((int64_t)256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    memoryorganizer_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(out, srcp0, srcp1, srcp2, width, height);
}

int64_t getTotalScaleSize(int64_t width, int64_t height){
    int64_t result = 0;
    for (int scale = 0; scale < 6; scale++){
        result += width*height;
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return result;
}

//expects packed linear RGB input. Beware that each src1_d, src2_d and temp_d must be of size "totalscalesize" even if the actual image is contained in a width*height format
double ssimu2GPUProcess(float3* src1_d, float3* src2_d, float3* temp_d, float3* pinned, int64_t width, int64_t height, GaussianHandle& gaussianhandle, int64_t maxshared, hipStream_t stream){
    const int64_t totalscalesize = getTotalScaleSize(width, height);

    //step 1 : fill the downsample part
    int64_t nw = width;
    int64_t nh = height;
    int64_t index = 0;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d+index, src1_d+index+nw*nh, nw, nh, stream);
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    index = 0;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src2_d+index, src2_d+index+nw*nh, nw, nh, stream);
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    rgb_to_positive_xyb(src1_d, totalscalesize, stream);
    rgb_to_positive_xyb(src2_d, totalscalesize, stream);

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    std::vector<float3> allscore_res;
    try{
        allscore_res = allscore_map(src1_d, src2_d, temp_d, pinned, width, height, maxshared, gaussianhandle, stream);
    } catch (const VshipError& e){
        throw e;
    }

    //step 6 : format the vector
    std::vector<float> measure_vec(108);

    for (int plane = 0; plane < 3; plane++){
        for (int scale = 0; scale < 6; scale++){
            for (int n = 0; n < 2; n++){
                for (int i = 0; i < 3; i++){
                    if (plane == 0) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].x;
                    if (plane == 1) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].y;
                    if (plane == 2) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].z;
                }
            }
        }
    }

    //step 7 : enjoy !
    const float ssim = final_score(measure_vec);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    return ssim;
}

class SSIMU2ComputingImplementation{
    float3* pinned;
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    int64_t width;
    int64_t height;
    int maxshared;
    hipStream_t stream;
public:
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2){
        hipStreamCreate(&stream);
        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, stream);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, stream);

        this->width = converter1.getWidth();
        this->height = converter1.getHeight();

        //assert they have the same width/height
        if (converter2.getWidth() != width || converter2.getHeight() != height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        gaussianhandle.init();

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;

        const int64_t pinnedsize = allocsizeScore(width, height, maxshared);
        hipError_t erralloc = hipHostMalloc(&pinned, sizeof(float3)*pinnedsize);
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
    double run(const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        const int64_t totalscalesize = getTotalScaleSize(width, height);

        float3* mem_d;
        hipError_t erralloc = hipMallocAsync(&mem_d, sizeof(float3)*totalscalesize*3, stream); //2 base image and 1 reduction+copy buffer
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        float3* src1_d = mem_d;
        float3* src2_d = mem_d + totalscalesize;
        float3* temp_d = mem_d + 2*totalscalesize;

        float* tempcolorconversionDST[3] = {((float*)temp_d), ((float*)temp_d)+width*height, ((float*)temp_d)+2*width*height};
        //first we convert our srcp1 input inside temp_d
        converter1.convert(tempcolorconversionDST, srcp1, lineSize);
        //then we convert it to float3 and put it where it belongs in src1_d
        memoryorganizer(src1_d, tempcolorconversionDST[0], tempcolorconversionDST[1], tempcolorconversionDST[2], width, height, stream);

        //same for srcp2
        converter2.convert(tempcolorconversionDST, srcp2, lineSize2);
        memoryorganizer(src2_d, tempcolorconversionDST[0], tempcolorconversionDST[1], tempcolorconversionDST[2], width, height, stream);

        double res;
        try {
            res = ssimu2GPUProcess(src1_d, src2_d, temp_d, pinned, width, height, gaussianhandle, maxshared, stream);
        } catch (const VshipError& e){
            hipFree(mem_d);
            throw e;
        }

        hipFreeAsync(mem_d, stream);

        return res;
    }
};

}