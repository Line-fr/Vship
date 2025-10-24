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
#include "pooling.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include "gaussianBlur.hpp"
#include "maskingModel.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing& temporalRing1, TemporalRing& temporalRing2, CSF_Handler& csfhandle, GaussianHandle& gaussianhandle, DisplayModel* model, int64_t maxshared, hipStream_t stream1, hipStream_t stream2, hipEvent_t event, hipEvent_t event2){
    int64_t width = temporalRing1.width;
    int64_t height = temporalRing1.height;

    int allocatedPlanes = 5;
    int stream1_supPlane = 0;
    //to fit the pyramid, we need 5/3 the normal size for safety
    int gaussianPyrSizeMultiplierNumerator = 5;
    int gaussianPyrSizeMultiplierDenominator = 3;

    const int64_t bandOffset = width*height*gaussianPyrSizeMultiplierNumerator/gaussianPyrSizeMultiplierDenominator;
    float* mem_d;
    hipError_t erralloc = hipMallocAsync(&mem_d, sizeof(float)*(allocatedPlanes+stream1_supPlane)*bandOffset, stream1);
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
    //now we only use stream1, this is the time for merging both sources
    
    int w = width;
    int h = height;
    const int levels = LPyr1.getSize();
    for (int band = 0; band < levels; band++){
        if (band != levels-1){
            for (int channel = 0; channel < 4; channel++){
                preGaussianPreCompute(LPyr1.getLbkg(band), LPyr1.getContrast(channel, band), LPyr2.getContrast(channel, band), w, h, channel, band, csfhandle, stream1);
            }
            //it writes on its source
            computeD(LPyr1.getContrast(0, band), LPyr1.getContrast(1, band), LPyr1.getContrast(2, band), LPyr1.getContrast(3, band), LPyr2.getContrast(0, band), LPyr2.getContrast(1, band),LPyr2.getContrast(2, band), LPyr2.getContrast(3, band), w, h, gaussianhandle, stream1);
        } else {
            //baseband
            for (int channel = 0; channel < 4; channel++){
                computeD_baseband(LPyr1.getLbkg(band), LPyr1.getContrast(channel, band), LPyr2.getContrast(channel, band), w, h, channel, band, csfhandle, stream1);
            }
        }
        w = (w+1)/2;
        h = (h+1)/2;
    }
    //stream2 waits for stream1 to be done before freeing memory (memory of Lpyr2)
    hipEventRecord(event2, stream1);
    hipStreamWaitEvent(stream2, event2);
    hipFreeAsync(mem_d2, stream2);
    //the complete full distortion map is stored in Lpyr1
    //and we can use Lpyr1 Lbkg place as temporary storage for pooling
    std::vector<float> scores(levels*4); //per_band and per channel
    int tempOffset = 0;
    float* temp = LPyr1.getLbkg(0); //we take all the space! so we use band0
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const auto [w, h] = LPyr1.getResolution(band);
            const int64_t size = w*h;
            computeMean<2>(LPyr1.getContrast(channel, band), temp+tempOffset, size, true, stream1);
            tempOffset++; //we stored the norm at temp[tempOffset]
        }
    }
    hipMemcpyDtoHAsync(scores.data(), temp, sizeof(float)*levels*4, stream1);
    hipStreamSynchronize(stream1);
    hipFreeAsync(mem_d, stream1);

    //we have our levels*4 scores
    float finalValue = 0;
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const float val = scores[band*4+channel];
            finalValue += std::pow(val, 4);
        }
    }
    finalValue = std::pow(finalValue/(float)(levels*4), 1./4.);

    return finalValue;
}

double toJOD(double a){
    if (a > 0.1){
        return 10. - jod_a * std::pow(a, jod_exp);
    } else {
        const double jod_a_p = jod_a * (std::pow(0.1, jod_exp-1.));
        return 10. - jod_a_p * a;
    }
}

//currently at 32 planes + 3*fps/2 planes consumed
class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    //std::vector<float> all_scores; //for past frames
    int numFrame = 0;
    double score_squareSum; //to avoid recomputing it all the time
    TemporalRing temporalRing1; //source
    TemporalRing temporalRing2; //encoded
    CSF_Handler csf_handler;
    GaussianHandle gaussianhandle;
    int64_t source_width = 0;
    int64_t source_height = 0;
    int64_t resize_width = 0;
    int64_t resize_height = 0;
    int maxshared = 0;
    hipStream_t stream1 = 0;
    hipStream_t stream2 = 0;
    hipEvent_t event;
    hipEvent_t event2;
public:
    void init(int64_t width, int64_t height, float fps, bool resizeToDisplay, std::string model_key){
        model = new DisplayModel(model_key);
        
        source_width = width;
        source_height = height;

        if (resizeToDisplay){
            //we resize to match width since display_width/width <= display_height/height
            if (width*model->resolution[1] <= height*model->resolution[0]){
                resize_width = model->resolution[0];
                resize_height = (height*model->resolution[0])/width;
            } else {
                //here we resize to match height
                resize_height = model->resolution[1];
                resize_width = (width*model->resolution[1])/height;
            }
        } else {
            resize_width = width;
            resize_height = height;
        }
        std::cout << "base/resize : " << width << "x" << height << "/" << resize_width << "x" << resize_height << std::endl;

        this->fps = fps;

        temporalRing1.init(fps, resize_width, resize_height);
        temporalRing2.init(fps, resize_width, resize_height);
        csf_handler.init(resize_width, resize_height, model->get_screen_ppd());
        gaussianhandle.init();

        hipStreamCreate(&stream1);
        hipStreamCreate(&stream2);
        hipEventCreate(&event);
        hipEventCreate(&event2);

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;
    }
    void destroy(){
        temporalRing1.destroy();
        temporalRing2.destroy();
        csf_handler.destroy();
        gaussianhandle.destroy();
        delete model;
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
        hipEventDestroy(event);
        hipEventDestroy(event2);
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

        const float Y_peak = model->max_luminance;
        const float Y_black = model->getBlackLevel();
        const float Y_refl = model->getReflLevel();
        const float exposure = model->exposure;

        //we put the frame's planes on GPU
        //do we write directly in final after stride eliminaation?
        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[0];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[0]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream1);
        if (is_resized) resizePlane(src1_d[0], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[1];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[1]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream1);
        if (is_resized) resizePlane(src1_d[1], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated = is_resized ? tempStrideEliminated : src1_d[2];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d, (void*)(srcp1[2]), stride * source_height, stream1));
        strideEliminator<T>(tempStrideEliminated, mem_d, stride, source_width, source_height, stream1);
        rgb_to_linrgb(tempStrideEliminated, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream1);
        if (is_resized) resizePlane(src1_d[2], tempResize, tempStrideEliminated, source_width, source_height, resize_width, resize_height, stream1);

        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[0];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[0]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(tempStrideEliminated2, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream2);
        if (is_resized) resizePlane(src2_d[0], tempResize2, tempStrideEliminated2, source_width, source_height, resize_width, resize_height, stream2);
        
        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[1];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[1]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(tempStrideEliminated2, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream2);
        if (is_resized) resizePlane(src2_d[1], tempResize2, tempStrideEliminated2, source_width, source_height, resize_width, resize_height, stream2);
        
        tempStrideEliminated2 = is_resized ? tempStrideEliminated2 : src2_d[2];
        GPU_CHECK(hipMemcpyHtoDAsync(mem_d2, (void*)(srcp2[2]), stride2 * source_height, stream2));
        strideEliminator<T>(tempStrideEliminated2, mem_d2, stride2, source_width, source_height, stream2);
        rgb_to_linrgb(tempStrideEliminated2, source_width*source_height, Y_peak, Y_black, Y_refl, exposure, stream2);
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
        const float current_score = CVVDPprocess(dstp, dststride, temporalRing1, temporalRing2, csf_handler, gaussianhandle, model, maxshared, stream1, stream2, event, event2);
        score_squareSum += std::pow(current_score, beta_t);
        float resQ;
        if (numFrame == 0){
            resQ = current_score * image_int;
        } else {
            resQ = std::pow(score_squareSum/(double)numFrame, 1./beta_t);
        }
        numFrame++;
        return toJOD(resQ);
    }
    //empties the history.
    void flushTemporalRing(){
        numFrame = 0;
        score_squareSum = 0.;
        temporalRing1.reset();
        temporalRing2.reset();
    }
};

}