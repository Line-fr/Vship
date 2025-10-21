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
#include "lpyr.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing temporalRing1, TemporalRing temporalRing2, DisplayModel* model, int64_t maxshared, hipStream_t stream1, hipStream_t stream2, hipEvent_t event){
    int64_t width = temporalRing1.width;
    int64_t height = temporalRing1.height;

    int allocatedPlanes = 5;
    //to fit the pyramid, we need 4/3 the normal size
    int gaussianPyrSizeMultiplierNumerator = 4;
    int gaussianPyrSizeMultiplierDenominator = 3;

    const int64_t bandOffset = width*height*gaussianPyrSizeMultiplierNumerator/gaussianPyrSizeMultiplierDenominator;
    float* mem_d;
    hipError_t erralloc = hipMallocAsync(&mem_d, sizeof(float)*allocatedPlanes*bandOffset, stream1);
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }
    //second stream
    float* mem_d2;
    erralloc = hipMallocAsync(&mem_d2, sizeof(float)*allocatedPlanes*bandOffset, stream2);
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }

    float* Y_sustained1 = mem_d;
    float* RG_sustained1 = mem_d + bandOffset;
    float* YV_sustained1 = mem_d + 2*bandOffset;
    float* Y_transient1 = mem_d + 3*bandOffset;

    float* Y_sustained2 = mem_d2;
    float* RG_sustained2 = mem_d2 + bandOffset;
    float* YV_sustained2 = mem_d2 + 2*bandOffset;
    float* Y_transient2 = mem_d2 + 3*bandOffset;

    //let's get the temporal channel out of the temporal ring!
    computeTemporalChannels(temporalRing1, Y_sustained1, RG_sustained1, YV_sustained1, Y_transient1, stream1);
    computeTemporalChannels(temporalRing2, Y_sustained2, RG_sustained2, YV_sustained2, Y_transient2, stream2);

    const float ppd = model->get_screen_ppd();
    LpyrManager LPyr1(mem_d, width, height, ppd, bandOffset, stream1);
    LpyrManager LPyr2(mem_d2, width, height, ppd, bandOffset, stream2);

    //stream1 waits for stream2 to be done
    hipEventRecord(event, stream2);
    hipStreamWaitEvent(stream1, event);

    hipStreamSynchronize(stream1);

    hipFreeAsync(mem_d, stream1);
    hipFreeAsync(mem_d2, stream2);

    return 10.;
}

//currently at 32 planes + 3*fps/2 planes consumed
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
    hipStream_t stream1 = 0;
    hipStream_t stream2 = 0;
    hipEvent_t event;
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

        hipStreamCreate(&stream1);
        hipStreamCreate(&stream2);
        hipEventCreate(&event);

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
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
        hipEventDestroy(event);
    }
    template <InputMemType T>
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int64_t stride, int64_t stride2){
        bool is_resized = (source_width != resize_width || source_height != resize_height);
        int64_t resizeBufferBytes;
        if (is_resized){
            resizeBufferBytes = resize_width*resize_height*sizeof(float) + source_width*source_height*sizeof(float);
        } else {
            resizeBufferBytes = 0;
        }
        //allocate memory to send the raw to gpu
        float* mem_d;
        hipError_t erralloc = hipMallocAsync(&mem_d, (std::max(stride, stride2)*source_height + resizeBufferBytes), stream1);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        //for second stream
        float* mem_d2;
        erralloc = hipMallocAsync(&mem_d2, (std::max(stride, stride2)*source_height + resizeBufferBytes), stream2);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }

        //defined onnly if is_resized is true
        float* tempResize = (float*)((uint8_t*)mem_d + std::max(stride, stride2)*source_height); //of size resize plane
        float* tempStrideEliminated = tempResize + resize_width*resize_height; //of size source plane (float)

        //for second stream
        float* tempResize2 = (float*)((uint8_t*)mem_d2 + std::max(stride, stride2)*source_height); //of size resize plane
        float* tempStrideEliminated2 = tempResize2 + resize_width*resize_height; //of size source plane (float)

        //free up a plane by increasing history to 1, we can now edit the frame 0 which is blank
        temporalRing1.rotate();
        temporalRing2.rotate();
        //take color planes from the ring
        float* source_ptr = temporalRing1.getFramePointer(0);
        float* encoded_ptr = temporalRing2.getFramePointer(0);
        float* src1_d[3] = {source_ptr, source_ptr+resize_width*resize_height, source_ptr+2*resize_width*resize_height};
        float* src2_d[3] = {encoded_ptr, encoded_ptr+resize_width*resize_height, encoded_ptr+2*resize_width*resize_height};

        //we put the frame's planes on GPU
        //do we write directly in final after stride eliminaation?
        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[0];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[0]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream1);
        if (is_resized) resizePlane(src1_d[0], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[1];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[1]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream1);
        if (is_resized) resizePlane(src1_d[1], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[2];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[2]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, stream1);
        if (is_resized) resizePlane(src1_d[2], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[0];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[0]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(tempStrideEliminated2, source_width*source_height, stream2);
        if (is_resized) resizePlane(src2_d[0], tempResize2, tempStrideEliminated2, source_width, source_height, resize_width, resize_height, stream2);
        
        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[1];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[1]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(is_resized ? tempStrideEliminated2 : src2_d[1], source_width*source_height, stream2);
        if (is_resized) resizePlane(src2_d[1], tempResize2, tempStrideEliminated2, source_width, source_height, resize_width, resize_height, stream2);
        
        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[2];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[2]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(tempStrideEliminated2, source_width*source_height, stream2);
        if (is_resized) resizePlane(src2_d[2], tempResize2, tempStrideEliminated2, source_width, source_height, resize_width, resize_height, stream2);

        //colorspace conversion
        linrgb_to_dkl(src1_d, resize_width*resize_height, stream1);
        linrgb_to_dkl(src2_d, resize_width*resize_height, stream2);

        hipFreeAsync(mem_d, stream1);
        hipFreeAsync(mem_d2, stream2);
        //and we are done, the frame is loaded in the ring with the right colorspace, next step is temporal filtering
    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        loadImageToRing<T>(srcp1, srcp2, stride, stride2);
        return CVVDPprocess(dstp, dststride, temporalRing1, temporalRing2, model, maxshared, stream1, stream2, event);
    }
    //empties the history.
    void flushTemporalRing(){
        temporalRing1.reset();
        temporalRing2.reset();
    }
};

}