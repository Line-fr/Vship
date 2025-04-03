#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/torgbs.hpp"
#include "../util/float3operations.hpp"
#include "../util/threadsafeset.hpp"
#include "makeXYB.hpp"
#include "downsample.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

namespace ssimu2{

double ssimu2process(const uint8_t *srcp1[3], const uint8_t *srcp2[3], float3* pinned, int stride, int width, int height, float* gaussiankernel, int maxshared, hipStream_t stream){

    int wh = width*height;
    int whs[6] = {wh, ((height-1)/2 + 1)*((width-1)/2 + 1), ((height-1)/4 + 1)*((width-1)/4 + 1), ((height-1)/8 + 1)*((width-1)/8 + 1), ((height-1)/16 + 1)*((width-1)/16 + 1), ((height-1)/32 + 1)*((width-1)/32 + 1)};
    int whs_integral[7];
    whs_integral[0] = 0;
    for (int i = 0; i < 7; i++){
        whs_integral[i+1] = whs_integral[i] + whs[i];
    }
    int totalscalesize = whs_integral[6];

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;

    float3* mem_d;
    erralloc = hipMallocAsync(&mem_d, sizeof(float3)*totalscalesize*(2 + 6), stream); //2 base image and 6 working buffers
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }

    float3* src1_d = mem_d; //length totalscalesize
    float3* src2_d = mem_d + totalscalesize;

    float3* temp_d = mem_d + 2*totalscalesize;
    float3* temps11_d = mem_d + 3*totalscalesize;
    float3* temps22_d = mem_d + 4*totalscalesize;
    float3* temps12_d = mem_d + 5*totalscalesize;
    float3* tempb1_d = mem_d + 6*totalscalesize;
    float3* tempb2_d = mem_d + 7*totalscalesize;

    uint8_t *memory_placeholder[3] = {(uint8_t*)temp_d, (uint8_t*)temp_d+stride*height, (uint8_t*)temp_d+2*stride*height};
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp1[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp1[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp1[2], stride * height, stream));
    memoryorganizer(src1_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp2[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp2[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp2[2], stride * height, stream));
    memoryorganizer(src2_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    rgb_to_linear(src1_d, totalscalesize, stream);
    rgb_to_linear(src2_d, totalscalesize, stream);

    //step 1 : fill the downsample part
    int nw = width;
    int nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d+whs_integral[scale-1], src1_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src2_d+whs_integral[scale-1], src2_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    rgb_to_positive_xyb(src1_d, totalscalesize, stream);
    rgb_to_positive_xyb(src2_d, totalscalesize, stream);

    //step 3 : fill buffer s11 s22 and s12 and blur everything
    multarray(src1_d, src1_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps11_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src1_d, src2_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps12_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src2_d, src2_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps22_d, totalscalesize, width, height, gaussiankernel, stream);

    gaussianBlur(src1_d, tempb1_d, totalscalesize, width, height, gaussiankernel, stream);
    gaussianBlur(src2_d, tempb2_d, totalscalesize, width, height, gaussiankernel, stream);

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    std::vector<float3> allscore_res;
    try{
        allscore_res = allscore_map(src1_d, src2_d, tempb1_d, tempb2_d, temps11_d, temps22_d, temps12_d, temp_d, pinned, width, height, maxshared, stream);
    } catch (const VshipError& e){
        hipFree(mem_d);
        throw e;
    }

    //we are done with the gpu at that point and the synchronization has already been done in allscore_map
    hipFreeAsync(mem_d, stream);

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

typedef struct Ssimulacra2Data{
    VSNode *reference;
    VSNode *distorted;
    float3** PinnedMemPool;
    int maxshared;
    float* gaussiankernel_d;
    hipStream_t* streams;
    int streamnum = 0;
    threadSet* streamSet;
} Ssimulacra2Data;

static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        int height = vsapi->getFrameHeight(src1, 0);
        int width = vsapi->getFrameWidth(src1, 0);
        int stride = vsapi->getStride(src1, 0);

        VSFrame *dst = vsapi->copyFrame(src2, core);

        const uint8_t *srcp1[3] = {
            vsapi->getReadPtr(src1, 0),
            vsapi->getReadPtr(src1, 1),
            vsapi->getReadPtr(src1, 2),
        };

        const uint8_t *srcp2[3] = {
            vsapi->getReadPtr(src2, 0),
            vsapi->getReadPtr(src2, 1),
            vsapi->getReadPtr(src2, 2),
        };

        double val;
        const int stream = d->streamSet->pop();
        try{
            val = ssimu2process(srcp1, srcp2, d->PinnedMemPool[stream], stride, width, height, d->gaussiankernel_d, d->maxshared, d->streams[stream]);
        } catch (const VshipError& e){
            vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
            d->streamSet->insert(stream);
            vsapi->freeFrame(src1);
            vsapi->freeFrame(src2);
            return NULL;
        }
        d->streamSet->insert(stream);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_SSIMULACRA2", val, maReplace);

        // Release the source frame
        vsapi->freeFrame(src1);
        vsapi->freeFrame(src2);

        // A reference is consumed when it is returned, so saving the dst reference somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return NULL;
}

// Free all allocated data on filter destruction
static void VS_CC ssimulacra2Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < d->streamnum; i++){
        hipHostFree(d->PinnedMemPool[i]);
        hipStreamDestroy(d->streams[i]);
    }
    free(d->PinnedMemPool);
    hipFree(d->gaussiankernel_d);
    free(d->streams);
    delete d->streamSet;
    //vsapi->setThreadCount(d->oldthreadnum, core);

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data d;
    Ssimulacra2Data *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi, true);
    d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi, true);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, VshipError(DifferingInputType, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.bitsPerSample != 16) || (viref->format.colorFamily != cfRGB) || viref->format.sampleType != stInteger){
        vsapi->mapSetError(out, VshipError(NonRGBSInput, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }

    try{
        //if succeed, this function also does hipSetDevice
        helper::gpuFullCheck(gpuid);
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    //hipSetDevice(gpuid);

    hipDeviceSetCacheConfig(hipFuncCachePreferNone);
    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);
    d.maxshared = devattr.sharedMemPerBlock;

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    VSCoreInfo infos;
    vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //size_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    //vsapi->setThreadCount(std::min((int)((float)(freemem - 20*(1llu << 20))/(8*sizeof(float3)*videowidth*videoheight*(1.33333))), d.oldthreadnum), core);

    d.streamnum = vsapi->mapGetInt(in, "numStream", 0, &error);
    if (error != peSuccess){
        d.streamnum = infos.numThreads;
    }

    d.streamnum = std::min(d.streamnum, infos.numThreads); // vs threads < numStream would make no sense
    d.streamnum = std::min(d.streamnum, (int)(devattr.totalGlobalMem/(32*4*viref->width*viref->height))); //VRAM overcommit partial protection
    d.streams = (hipStream_t*)malloc(sizeof(hipStream_t)*d.streamnum);
    for (int i = 0; i < d.streamnum; i++){
        hipStreamCreate(d.streams + i);
    }

    std::set<int> newstreamset;
    for (int i = 0; i < d.streamnum; i++){
        newstreamset.insert(i);
    }
    d.streamSet = new threadSet(newstreamset);

    const int pinnedsize = allocsizeScore(viref->width, viref->height, d.maxshared);
    d.PinnedMemPool = (float3**)malloc(sizeof(float3*)*d.streamnum);
    hipError_t erralloc;
    for (int i = 0; i < d.streamnum; i++){
        erralloc = hipHostMalloc(d.PinnedMemPool+i, sizeof(float3)*pinnedsize);
        if (erralloc != hipSuccess){
            vsapi->mapSetError(out, VshipError(OutOfRAM, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d.reference);
            vsapi->freeNode(d.distorted);
            return;
        }
    }

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < d.streamnum; i++){
        data->streams[i] = d.streams[i];
    }

    float gaussiankernel[2*GAUSSIANSIZE+1];
    for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
        gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(TAU*SIGMA*SIGMA));
    }

    hipMalloc(&(data->gaussiankernel_d), sizeof(float)*(2*GAUSSIANSIZE+1));
    hipMemcpyHtoD((hipDeviceptr_t)data->gaussiankernel_d, gaussiankernel, (2*GAUSSIANSIZE+1)*sizeof(float));

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}

}