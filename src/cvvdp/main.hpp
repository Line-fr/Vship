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

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing temporalRing1, TemporalRing temporalRing2, int64_t width, int64_t height, int64_t maxshared, hipStream_t stream){
    
    return 10.;
}

class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    TemporalRing temporalRing1; //source
    TemporalRing temporalRing2; //encoded
    int64_t width = 0;
    int64_t height = 0;
    int maxshared = 0;
    hipStream_t stream = 0;
public:
    void init(int64_t width, int64_t height, float fps, std::string model_key){
        this->width = width;
        this->height = height;
        this->fps = fps;

        temporalRing1.init(fps, width, height);
        temporalRing2.init(fps, width, height);
        model = new DisplayModel(model_key);

        hipStreamCreate(&stream);

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;
    }
    void destroy(){
        temporalRing1.destroy();
        temporalRing2.destroy();
        delete model;
        hipStreamDestroy(stream);
    }
    template <InputMemType T>
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int64_t stride, int64_t stride2){
        //allocate memory to send the raw to gpu
        float* mem_d;
        hipError_t erralloc = hipMallocAsync(&mem_d, std::max(stride, stride2)*height, stream);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }

        //free up a plane by increasing history to 1, we can now edit the frame 0 which is blank
        temporalRing1.rotate();
        temporalRing2.rotate();
        //take color planes from the ring
        float* source_ptr = temporalRing1.getFramePointer(0);
        float* encoded_ptr = temporalRing2.getFramePointer(0);
        float* src1_d[3] = {source_ptr, source_ptr+width*height, source_ptr+2*width*height};
        float* src2_d[3] = {encoded_ptr, encoded_ptr+width*height, encoded_ptr+2*width*height};

        //we put the frame's planes on GPU
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[0]), stride * height, stream));
        strideEliminator<T>(src1_d[0], mem_d, stride, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[1]), stride * height, stream));
        strideEliminator<T>(src1_d[1], mem_d, stride, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[2]), stride * height, stream));
        strideEliminator<T>(src1_d[2], mem_d, stride, width, height, stream);

        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[0]), stride2 * height, stream));
        strideEliminator<T>(src2_d[0], mem_d, stride2, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[1]), stride2 * height, stream));
        strideEliminator<T>(src2_d[1], mem_d, stride2, width, height, stream);
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[2]), stride2 * height, stream));
        strideEliminator<T>(src2_d[2], mem_d, stride2, width, height, stream);

        hipFreeAsync(mem_d, stream); //we are done sending frames to GPU

        //colorspace conversion
        rgb_to_dkl(src1_d, width*height, stream);
        rgb_to_dkl(src2_d, width*height, stream);

        //and we are done, the frame is loaded in the ring with the right colorspace, next step is temporal filtering
    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        loadImageToRing<T>(srcp1, srcp2, stride, stride2);
        return CVVDPprocess(dstp, dststride, temporalRing1, temporalRing2, width, height, maxshared, stream);
    }
};

}