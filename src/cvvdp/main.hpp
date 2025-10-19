#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"

#include "parameters.hpp"
#include "display_models.hpp"
#include "colors.hpp"
#include "temporalFilter.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing temporalRing, int64_t width, int64_t height, int64_t maxshared, hipStream_t stream){
    
    return 10.;
}

class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    TemporalRing tempFilter;
    int64_t width = 0;
    int64_t height = 0;
    int maxshared = 0;
    hipStream_t stream = 0;
public:
    void init(int64_t width, int64_t height, float fps, std::string model_key){
        this->width = width;
        this->height = height;
        this->fps = fps;

        tempFilter.init(fps, width, height);
        model = new DisplayModel(model_key);

        hipStreamCreate(&stream);

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;
    }
    void destroy(){
        tempFilter.destroy();
        delete model;
        hipStreamDestroy(stream);
    }
    template <InputMemType T>
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int64_t stride, int64_t stride2){
        float* mem_d;
        hipError_t erralloc = hipMallocAsync(&mem_d, std::max(stride, stride2)*height+6*sizeof(float)*width*height, stream); //max just in case stride is ridiculously large
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        //initial color planes
        float* src1_d[3] = {mem_d, mem_d+width*height, mem_d+2*width*height};
        float* src2_d[3] = {mem_d+3*width*height, mem_d+4*width*height, mem_d+5*width*height};

        //we put the frame's planes on GPU
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[0]), stride * height, stream));
        strideEliminator<T>(src1_d[0], mem_d+6*width*height, stride, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[1]), stride * height, stream));
        strideEliminator<T>(src1_d[1], mem_d+6*width*height, stride, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[2]), stride * height, stream));
        strideEliminator<T>(src1_d[2], mem_d+6*width*height, stride, width, height, stream);

        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[0]), stride2 * height, stream));
        strideEliminator<T>(src2_d[0], mem_d+6*width*height, stride2, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[1]), stride2 * height, stream));
        strideEliminator<T>(src2_d[1], mem_d+6*width*height, stride2, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[2]), stride2 * height, stream));
        strideEliminator<T>(src2_d[2], mem_d+6*width*height, stride2, width, height, stream);

        //colorspace conversion
        rgb_to_dkl(src1_d, width*height, stream);
        rgb_to_dkl(src2_d, width*height, stream);

        hipFreeAsync(mem_d, stream);
    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        loadImageToRing<T>(srcp1, srcp2, stride, stride2);
        return CVVDPprocess(dstp, dststride, tempFilter, width, height, maxshared, stream);
    }
};

}