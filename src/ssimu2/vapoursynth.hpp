#pragma once

#include "../util/torgbs.hpp"
#include "main.hpp"

namespace ssimu2{

typedef struct Ssimulacra2Data{
    VSNode *reference;
    VSNode *distorted;
    SSIMU2ComputingImplementation* ssimu2Streams;
    int streamnum = 0;
    threadSet<int>* streamSet;
} Ssimulacra2Data;

static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;
    
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        //int64_t height = vsapi->getFrameHeight(src1, 0);
        //int64_t width = vsapi->getFrameWidth(src1, 0);

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

        const int64_t lineSize[3] = {
            vsapi->getStride(src1, 0),
            vsapi->getStride(src1, 1),
            vsapi->getStride(src1, 2),
        };
        const int64_t lineSize2[3] = {
            vsapi->getStride(src2, 0),
            vsapi->getStride(src2, 1),
            vsapi->getStride(src2, 2),
        };

        double val;
        const int stream = d->streamSet->pop();
        SSIMU2ComputingImplementation& ssimu2Stream = d->ssimu2Streams[stream];
        try{
            val = ssimu2Stream.run(srcp1, srcp2, lineSize, lineSize2);
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
    (void)core;
    
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < d->streamnum; i++){
        d->ssimu2Streams[i].destroy();
    }
    free(d->ssimu2Streams);
    delete d->streamSet;

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void * userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    Ssimulacra2Data d;
    Ssimulacra2Data *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi, false);
    d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi, false);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

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

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    VSCoreInfo infos;
    vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //int64_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    d.streamnum = vsapi->mapGetInt(in, "numStream", 0, &error);
    if (error != peSuccess){
        d.streamnum = infos.numThreads;
    }

    d.streamnum = std::min(d.streamnum, infos.numThreads); // vs threads < numStream would make no sense
    d.streamnum = std::max(d.streamnum, 1); //at least one stream to not just wait indefinitely

    Vship_Colorspace_t src_colorspace; //vapoursynth handles the conversion, this is what we get from vs
    src_colorspace.width = viref->width;
    src_colorspace.target_width = -1;
    src_colorspace.height = viref->height;
    src_colorspace.target_height = -1;
    src_colorspace.crop = {0, 0, 0, 0};
    src_colorspace.sample = Vship_SampleFLOAT;
    src_colorspace.range = Vship_RangeFull;
    src_colorspace.subsampling = {0, 0};
    src_colorspace.colorFamily = Vship_ColorRGB;
    src_colorspace.YUVMatrix = Vship_MATRIX_RGB;
    src_colorspace.transferFunction = Vship_TRC_BT709;
    src_colorspace.primaries = Vship_PRIMARIES_BT709;

    try{
        d.ssimu2Streams = (SSIMU2ComputingImplementation*)malloc(sizeof(SSIMU2ComputingImplementation)*d.streamnum);
        for (int i = 0; i < d.streamnum; i++){
            d.ssimu2Streams[i].init(src_colorspace, src_colorspace);
        }
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    std::set<int> newstreamset;
    for (int i = 0; i < d.streamnum; i++){
        newstreamset.insert(i);
    }
    d.streamSet = new threadSet(newstreamset);

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}

}