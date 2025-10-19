#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"

#include "parameters.hpp"
#include "display_models.hpp"
#include "colors.hpp"
#include "resize.hpp"
#include "temporalFilter.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing temporalRing1, TemporalRing temporalRing2, DisplayModel* model, int64_t maxshared, hipStream_t stream){
    //int64_t width = temporalRing1.width;
    //int64_t height = temporalRing1.height;


    return 10.;
}

class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    TemporalRing temporalRing1; //source
    TemporalRing temporalRing2; //encoded
    int64_t source_width = 0;
    int64_t source_height = 0;
    int64_t resize_width = 0;
    int64_t resize_height = 0;
    int maxshared = 0;
    hipStream_t stream = 0;
public:
    void init(int64_t width, int64_t height, float fps, bool resizeToDisplay, std::string model_key){
        model = new DisplayModel(model_key);
        
        source_width = width;
        source_height = height;

        if (resizeToDisplay){
            //we resize to match width since display_width/width <= display_height/height
            if (width*model->resolution[1] <= height*model->resolution[0]){
                resize_width = width;
                resize_height = (height*model->resolution[0])/width;
            } else {
                //here we resize to match height
                resize_height = height;
                resize_width = (width*model->resolution[1])/height;
            }
        } else {
            resize_width = width;
            resize_height = height;
        }

        this->fps = fps;

        temporalRing1.init(fps, resize_width, resize_height);
        temporalRing2.init(fps, resize_width, resize_height);

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
        hipError_t erralloc = hipMallocAsync(&mem_d, std::max(stride, stride2)*source_height + resize_width*resize_height*sizeof(float) + source_width*source_height*sizeof(float), stream);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        float* tempResize = (float*)((uint8_t*)mem_d + std::max(stride, stride2)*source_height); //of size resize plane
        float* tempStrideEliminated = tempResize + resize_width*resize_height; //of size source plane (float)

        //free up a plane by increasing history to 1, we can now edit the frame 0 which is blank
        temporalRing1.rotate();
        temporalRing2.rotate();
        //take color planes from the ring
        float* source_ptr = temporalRing1.getFramePointer(0);
        float* encoded_ptr = temporalRing2.getFramePointer(0);
        float* src1_d[3] = {source_ptr, source_ptr+resize_width*resize_height, source_ptr+2*resize_width*resize_height};
        float* src2_d[3] = {encoded_ptr, encoded_ptr+resize_width*resize_height, encoded_ptr+2*resize_width*resize_height};

        //we put the frame's planes on GPU
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[0]), stride * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src1_d[0], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);
        
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[1]), stride * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src1_d[1], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);

        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[2]), stride * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src1_d[2], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);

        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[0]), stride2 * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride2, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src2_d[0], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);
        
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[1]), stride2 * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride2, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src2_d[1], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);
        
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp2[2]), stride2 * source_height, stream));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride2, source_width, source_height, stream);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream);
        resizePlane(src2_d[2], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream);

        //colorspace conversion
        linrgb_to_dkl(src1_d, resize_width*resize_height, stream);
        linrgb_to_dkl(src2_d, resize_width*resize_height, stream);

        hipFreeAsync(mem_d, stream);
        //and we are done, the frame is loaded in the ring with the right colorspace, next step is temporal filtering
    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        loadImageToRing<T>(srcp1, srcp2, stride, stride2);
        return CVVDPprocess(dstp, dststride, temporalRing1, temporalRing2, model, maxshared, stream);
    }
};

}