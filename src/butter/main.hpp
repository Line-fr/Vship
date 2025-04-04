#include <string>

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/torgbs.hpp"
#include "../util/float3operations.hpp"
#include "../util/threadsafeset.hpp"
#include "gaussianblur.hpp" 
#include "downupsample.hpp"
#include "Planed.hpp" //Plane_d class
#include "colors.hpp" //OpsinDynamicsImage
#include "separatefrequencies.hpp"
#include "maltaDiff.hpp"
#include "simplerdiff.hpp" //L2 +asym diff + same noise diff
#include "maskPsycho.hpp"
#include "combineMasks.hpp"
#include "diffnorms.hpp" //takes diffmap and returns norm2, norm3 and norminf

namespace butter{

Plane_d getdiffmap(Plane_d* src1_d, Plane_d* src2_d, float* mem_d, int width, int height, float intensity_multiplier, int maxshared, GaussianHandle& gaussianHandle, hipStream_t stream){
    //temporary planes
    Plane_d temp[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+1*width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};
    Plane_d temp2[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};

    //Psycho Image planes
    Plane_d lf1[3] = {Plane_d(mem_d+6*width*height, width, height, stream), Plane_d(mem_d+7*width*height, width, height, stream), Plane_d(mem_d+8*width*height, width, height, stream)};
    Plane_d mf1[3] = {Plane_d(mem_d+9*width*height, width, height, stream), Plane_d(mem_d+10*width*height, width, height, stream), Plane_d(mem_d+11*width*height, width, height, stream)};
    Plane_d hf1[2] = {Plane_d(mem_d+12*width*height, width, height, stream), Plane_d(mem_d+13*width*height, width, height, stream)};
    Plane_d uhf1[2] = {Plane_d(mem_d+14*width*height, width, height, stream), Plane_d(mem_d+15*width*height, width, height, stream)};

    Plane_d lf2[3] = {Plane_d(mem_d+16*width*height, width, height, stream), Plane_d(mem_d+17*width*height, width, height, stream), Plane_d(mem_d+18*width*height, width, height, stream)};
    Plane_d mf2[3] = {Plane_d(mem_d+19*width*height, width, height, stream), Plane_d(mem_d+20*width*height, width, height, stream), Plane_d(mem_d+21*width*height, width, height, stream)};
    Plane_d hf2[2] = {Plane_d(mem_d+22*width*height, width, height, stream), Plane_d(mem_d+23*width*height, width, height, stream)};
    Plane_d uhf2[2] = {Plane_d(mem_d+24*width*height, width, height, stream), Plane_d(mem_d+25*width*height, width, height, stream)};

    //to XYB
    opsinDynamicsImage(src1_d, temp, temp2[0], gaussianHandle, intensity_multiplier);
    opsinDynamicsImage(src2_d, temp, temp2[0], gaussianHandle, intensity_multiplier);
    GPU_CHECK(hipGetLastError());

    separateFrequencies(src1_d, temp, lf1, mf1, hf1, uhf1, gaussianHandle);
    separateFrequencies(src2_d, temp, lf2, mf2, hf2, uhf2, gaussianHandle);

    //no more needs for src1_d and src2_d so we reuse them as masks for butter
    Plane_d* block_diff_dc = src1_d; //size 3
    Plane_d* block_diff_ac = src2_d; //size 3

    //set the accumulators to 0
    for (int c = 0; c < 3; c++){
        block_diff_ac[c].fill0();
        block_diff_dc[c].fill0();
    }

    const float hf_asymmetry_ = 0.8f;

    const float wUhfMalta = 1.10039032555f;
    const float norm1Uhf = 71.7800275169f;
    MaltaDiffMap(uhf1[1].mem_d, uhf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wUhfMalta * hf_asymmetry_, wUhfMalta / hf_asymmetry_, norm1Uhf, stream);

    const float wUhfMaltaX = 173.5f;
    const float norm1UhfX = 5.0f;
    MaltaDiffMap(uhf1[0].mem_d, uhf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wUhfMaltaX * hf_asymmetry_, wUhfMaltaX / hf_asymmetry_, norm1UhfX, stream);

    const float wHfMalta = 18.7237414387f;
    const float norm1Hf = 4498534.45232f;
    MaltaDiffMapLF(hf1[1].mem_d, hf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wHfMalta * std::sqrt(hf_asymmetry_), wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf, stream);

    const float wHfMaltaX = 6923.99476109f;
    const float norm1HfX = 8051.15833247f;
    MaltaDiffMapLF(hf1[0].mem_d, hf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wHfMaltaX * std::sqrt(hf_asymmetry_), wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX, stream);

    const float wMfMalta = 37.0819870399f;
    const float norm1Mf = 130262059.556f;
    MaltaDiffMapLF(mf1[1].mem_d, mf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wMfMalta, wMfMalta, norm1Mf, stream);

    const float wMfMaltaX = 8246.75321353f;
    const float norm1MfX = 1009002.70582f;
    MaltaDiffMapLF(mf1[0].mem_d, mf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wMfMaltaX, wMfMaltaX, norm1MfX, stream);

    const float wmul[9] = {
      400.0f,         1.50815703118f,  0.0f,
      2150.0f,        10.6195433239f,  16.2176043152f,
      29.2353797994f, 0.844626970982f, 0.703646627719f,
    };

    //const float maxclamp = 85.7047444518;
    //const float kSigmaHfX = 10.6666499623;
    //const float w = 884.809801415;
    //sameNoiseLevels(hf1[1], hf2[1], block_diff_ac[1], temp[0], temp[1], kSigmaHfX, w, maxclamp, gaussiankernel_dmem);

    for (int c = 0; c < 3; c++){
        if (c < 2){
            L2AsymDiff(hf1[c].mem_d, hf2[c].mem_d, block_diff_ac[c].mem_d, width*height, wmul[c] * hf_asymmetry_, wmul[c] / hf_asymmetry_, stream);
        }
        L2diff(mf1[c].mem_d, mf2[c].mem_d, block_diff_ac[c].mem_d, width*height, wmul[3 + c], stream);
        L2diff(lf1[c].mem_d, lf2[c].mem_d, block_diff_dc[c].mem_d, width*height, wmul[6 + c], stream);
    }

    //from now on, lf and mf are not used so we will reuse the memory
    Plane_d mask = temp[1];
    Plane_d* temp3 = lf2;
    Plane_d* temp4 = mf2;

    MaskPsychoImage(hf1, uhf1, hf2, uhf2, temp3[0], temp4[0], mask, block_diff_ac, gaussianHandle);
    //at this point hf and uhf cannot be used anymore (they have been invalidated by the function)

    Plane_d diffmap = temp[0]; //we only need one plane
    computeDiffmap(mask.mem_d, block_diff_dc[0].mem_d, block_diff_dc[1].mem_d, block_diff_dc[2].mem_d, block_diff_ac[0].mem_d, block_diff_ac[1].mem_d, block_diff_ac[2].mem_d, diffmap.mem_d, width*height, stream);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!
    //GPU_CHECK(hipGetLastError());

    //printf("End result: %f, %f and %f\n", norm2, norm3, norminf);
    
    return diffmap;
}

std::tuple<float, float, float> butterprocess(const uint8_t *dstp, int dststride, const uint8_t *srcp1[3], const uint8_t *srcp2[3], float* mem_d, float* pinned, GaussianHandle& gaussianHandle, int stride, int width, int height, float intensity_multiplier, int maxshared, hipStream_t stream){
    int wh = width*height;
    const int totalscalesize = wh;

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used

    const int totalplane = 34;
    //initial color planes
    Plane_d src1_d[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};
    Plane_d src2_d[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};

    //we put the frame's planes on GPU
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[0]), stride * height, stream));
    src1_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[1]), stride * height, stream));
    src1_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[2]), stride * height, stream));
    src1_d[2].strideEliminator(mem_d+6*width*height, stride);

    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[0]), stride * height, stream));
    src2_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[1]), stride * height, stream));
    src2_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[2]), stride * height, stream));
    src2_d[2].strideEliminator(mem_d+6*width*height, stride);

    //computing downscaled before we overwrite src in getdiffmap (it s better for memory)
    int nwidth = (width-1)/2+1;
    int nheight = (height-1)/2+1;
    float* nmem_d = mem_d+6*width*height; //allow usage up to mem_d+8*width*height;
    Plane_d nsrc1_d[3] = {Plane_d(nmem_d, nwidth, nheight, stream), Plane_d(nmem_d+nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+2*nwidth*nheight, nwidth, nheight, stream)};
    Plane_d nsrc2_d[3] = {Plane_d(nmem_d+3*nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+4*nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+5*nwidth*nheight, nwidth, nheight, stream)};

    //we need to convert to linear rgb before downsampling
    linearRGB(src1_d);
    linearRGB(src2_d);

    //using 6 smaller planes is equivalent to 1.5 standard planes, so it fits within the 2 planes given here!)
    for (int i = 0; i < 3; i++){
        downsample(src1_d[i].mem_d, nsrc1_d[i].mem_d, width, height, stream);
        downsample(src2_d[i].mem_d, nsrc2_d[i].mem_d, width, height, stream);
    }

    Plane_d diffmap = getdiffmap(src1_d, src2_d, mem_d+8*width*height, width, height, intensity_multiplier, maxshared, gaussianHandle, stream);
    //diffmap is stored at mem_d+8*width*height so we can build after that the second smaller scale
    //smaller scale now
    nmem_d = mem_d+9*width*height;
    Plane_d diffmapsmall = getdiffmap(nsrc1_d, nsrc2_d, nmem_d+6*nwidth*nheight, nwidth, nheight, intensity_multiplier, maxshared, gaussianHandle, stream);

    addsupersample2X(diffmap.mem_d, diffmapsmall.mem_d, width, height, 0.5f, stream);

    //diffmap is in its final form
    if (dstp != NULL){
        diffmap.strideAdder(nmem_d, dststride);
        GPU_CHECK(hipMemcpyDtoHAsync((void*)(dstp), nmem_d, dststride * height, stream));
    }

    std::tuple<float, float, float> finalres;
    try{
        finalres = diffmapscore(diffmap.mem_d, mem_d+9*width*height, mem_d+10*width*height, pinned, width*height, stream);
    } catch (const VshipError& e){
        hipFree(mem_d);
        throw e;
    }

    return finalres;
}

typedef struct ButterData{
    VSNode *reference;
    VSNode *distorted;
    float intensity_multiplier;
    float** PinnedMemPool;
    float** VRAMMemPool;
    GaussianHandle gaussianHandle;
    int maxshared;
    int diffmap;
    hipStream_t* streams;
    int streamnum = 0;
    threadSet* streamSet;
} ButterData;

static const VSFrame *VS_CC butterGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    ButterData *d = (ButterData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        int height = vsapi->getFrameHeight(src1, 0);
        int width = vsapi->getFrameWidth(src1, 0);
        int stride = vsapi->getStride(src1, 0);

        VSFrame *dst;
        if (d->diffmap){
            VSVideoFormat formatout;
            vsapi->queryVideoFormat(&formatout, cfGray, stFloat, 32, 0, 0, core);
            dst = vsapi->newVideoFrame(&formatout, width, height, NULL, core);
        } else {
            dst = vsapi->copyFrame(src2, core);
        }

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
        
        std::tuple<float, float, float> val;
        
        const int stream = d->streamSet->pop();
        try{
            if (d->diffmap){
                val = butterprocess(vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), srcp1, srcp2, d->VRAMMemPool[stream], d->PinnedMemPool[stream], d->gaussianHandle, stride, width, height, d->intensity_multiplier, d->maxshared, d->streams[stream]);
            } else {
                val = butterprocess(NULL, 0, srcp1, srcp2, d->VRAMMemPool[stream], d->PinnedMemPool[stream], d->gaussianHandle, stride, width, height, d->intensity_multiplier, d->maxshared, d->streams[stream]);
            }
        } catch (const VshipError& e){
            vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
            d->streamSet->insert(stream);
            vsapi->freeFrame(src1);
            vsapi->freeFrame(src2);
            return NULL;
        }
        d->streamSet->insert(stream);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_2Norm", std::get<0>(val), maReplace);
        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_3Norm", std::get<1>(val), maReplace);
        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_INFNorm", std::get<2>(val), maReplace);

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
static void VS_CC butterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ButterData *d = (ButterData *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < d->streamnum; i++){
        hipFree(d->VRAMMemPool[i]);
        hipHostFree(d->PinnedMemPool[i]);
        hipStreamDestroy(d->streams[i]);
    }
    free(d->VRAMMemPool);
    free(d->PinnedMemPool);
    free(d->streams);
    d->gaussianHandle.destroy();
    delete d->streamSet;
    //vsapi->setThreadCount(d->oldthreadnum, core);

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC butterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    ButterData d;
    ButterData *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi);
    d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);
    VSVideoFormat formatout;
    vsapi->queryVideoFormat(&formatout, cfGray, stFloat, 32, 0, 0, core);
    VSVideoInfo viout = *viref;

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, VshipError(DifferingInputType, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.bitsPerSample != 32) || (viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, VshipError(NonRGBSInput, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int error;
    d.intensity_multiplier = vsapi->mapGetFloat(in, "intensity_multiplier", 0, &error);
    if (error != peSuccess){
        d.intensity_multiplier = 80.0f;
    }
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }
    d.diffmap = vsapi->mapGetInt(in, "distmap", 0, &error);
    if (error != peSuccess){
        d.diffmap = 0.;
    }

    if (d.diffmap){
        viout.format = formatout;
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

    try {
        d.gaussianHandle.init();
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    d.streamnum = std::min(d.streamnum, infos.numThreads);
    d.streamnum = std::min(d.streamnum, (int)(devattr.totalGlobalMem/(34*4*viref->width*viref->height))); //VRAM overcommit partial protection.
    d.streams = (hipStream_t*)malloc(sizeof(hipStream_t)*d.streamnum);
    for (int i = 0; i < d.streamnum; i++){
        hipStreamCreate(d.streams + i);
    }

    std::set<int> newstreamset;
    for (int i = 0; i < d.streamnum; i++){
        newstreamset.insert(i);
    }
    d.streamSet = new threadSet(newstreamset);

    const int pinnedsize = allocsizeScore(viref->width, viref->height);
    const int vramsize = viref->width*viref->height;
    d.PinnedMemPool = (float**)malloc(sizeof(float*)*d.streamnum);
    d.VRAMMemPool = (float**)malloc(sizeof(float*)*d.streamnum);
    hipError_t erralloc;
    for (int i = 0; i < d.streamnum; i++){
        erralloc = hipHostMalloc(d.PinnedMemPool+i, sizeof(float)*pinnedsize);
        if (erralloc != hipSuccess){
            vsapi->mapSetError(out, VshipError(OutOfRAM, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d.reference);
            vsapi->freeNode(d.distorted);
            return;
        }
        erralloc = hipMalloc(d.VRAMMemPool+i, sizeof(float)*34*vramsize);
        if (erralloc != hipSuccess){
            vsapi->mapSetError(out, VshipError(OutOfVRAM, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d.reference);
            vsapi->freeNode(d.distorted);
            return;
        }
    }

    data = (ButterData *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < d.streamnum; i++){
        data->streams[i] = d.streams[i];
    }
    data->maxshared = devattr.sharedMemPerBlock;

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "vship", &viout, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
}

}
