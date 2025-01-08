#include "../util/preprocessor.hpp"
#include "../util/float3operations.hpp"
#include "gaussianblur.hpp"

namespace butter{

class Plane_d{
public:
    int width, height;
    float* mem_d; //must be of size >= sizeof(float)*width*height;
    hipStream_t stream;
    Plane_d(float* mem_d, int width, int height, hipStream_t stream){
        this->mem_d = mem_d;
        this->height = height;
        this->width = width;
        this->stream = stream;
    }
    Plane_d(float* mem_d, Plane_d src){
        this->mem_d = mem_d;
        width = src.width;
        height = src.height;
        stream = src.stream;
        hipMemcpyDtoDAsync(mem_d, src.mem_d, sizeof(float)*width*height, stream);
    }
    void blur(Plane_d temp, float sigma, float border_ratio, float* gaussiankernel){
        const int gaussiansize = (int)(sigma * 5);
        loadGaussianKernel<<<dim3(1), dim3(2*gaussiansize+1), 0, stream>>>(gaussiankernel, gaussiansize, sigma);

        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        float weight_no_border = 0;
        for (int i = 0; i < 2*gaussiansize+1; i++){
            weight_no_border += gaussiankernel[i];
        }
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, temp.mem_d, width, height, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, mem_d, height, width, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
    }
    void blur(Plane_d dst, Plane_d temp, float sigma, float border_ratio, float* gaussiankernel){
        const int gaussiansize = (int)(sigma * 5);
        loadGaussianKernel<<<dim3(1), dim3(2*gaussiansize+1), 0, stream>>>(gaussiankernel, gaussiansize, sigma);

        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        float weight_no_border = 0;
        for (int i = 0; i < 2*gaussiansize+1; i++){
            weight_no_border += gaussiankernel[i];
        }
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, temp.mem_d, width, height, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp.mem_d, dst.mem_d, height, width, border_ratio, weight_no_border, gaussiankernel, gaussiansize);
    }
    void strideEliminator(float* strided, int stride){
        int wh = width*height;
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;
        strideEliminator_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, strided, stride, width, height);
    }
};

double butterprocess(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int stride, int width, int height, int maxshared, hipStream_t stream){
    
    int wh = width*height;
    const int totalscalesize = wh;

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;
    int tries = 10;

    const int gaussiantotal = 1024;
    const int totalplane = 24;
    float* mem_d;
    erralloc = hipMalloc(&mem_d, sizeof(float)*totalscalesize*(totalplane) + sizeof(float)*gaussiantotal); //2 base image and 6 working buffers
    while (erralloc != hipSuccess){
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); //0.5s with 10 tries -> shut down after 5 seconds of failing
        erralloc = hipMalloc(&mem_d, sizeof(float)*totalscalesize*(totalplane) + sizeof(float)*gaussiantotal); //2 base image and 6 working buffers
        tries--;
        if (tries <= 0){
            printf("ERROR, could not allocate VRAM for a frame, try lowering the number of vapoursynth threads\n");
            return -10000.;
        }
    }
    //GPU_CHECK(hipGetLastError());

    Plane_d src1_d[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};
    Plane_d src2_d[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};
    Plane_d temp[3] = {Plane_d(mem_d+6*width*height, width, height, stream), Plane_d(mem_d+7*width*height, width, height, stream), Plane_d(mem_d+8*width*height, width, height, stream)};

    float* gaussiankernel_dmem = (mem_d + totalplane*totalscalesize);

    hipEvent_t event_d;
    hipEventCreate(&event_d);

    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp1[0], stride * height, stream));
    src1_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp1[1], stride * height, stream));
    src1_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp1[2], stride * height, stream));
    src1_d[2].strideEliminator(mem_d+6*width*height, stride);

    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp2[0], stride * height, stream));
    src2_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp2[1], stride * height, stream));
    src2_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)srcp2[2], stride * height, stream));
    src2_d[2].strideEliminator(mem_d+6*width*height, stride);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    hipFree(mem_d);
    
    return 0.;
}

typedef struct {
    VSNode *reference;
    VSNode *distorted;
    int maxshared;
    hipStream_t streams[STREAMNUM];
    int oldthreadnum;
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

        const double val = butterprocess(srcp1, srcp2, stride, width, height, d->maxshared, d->streams[n%STREAMNUM]);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI", val, maReplace);

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

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamDestroy(d->streams[i]);
    }
    //vsapi->setThreadCount(d->oldthreadnum, core);

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC butterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    ButterData d;
    ButterData *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = vsapi->mapGetNode(in, "reference", 0, 0);
    d.distorted = vsapi->mapGetNode(in, "distorted", 0, 0);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, "BUTTERAUGLI: both clips must have the same format and dimensions");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, "BUTTERAUGLI: only works with RGBS format");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int count;
    if (hipGetDeviceCount(&count) != 0){
        vsapi->mapSetError(out, "could not detect devices, check gpu permissions\n");
    };
    if (count == 0){
        vsapi->mapSetError(out, "No GPU was found on the system for a given compilation type. Try switch nvidia/amd binary\n");
    }

    hipDeviceSetCacheConfig(hipFuncCachePreferNone);
    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    //VSCoreInfo infos;
    //vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //size_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    //vsapi->setThreadCount(std::min((int)((float)(freemem - 20*(1llu << 20))/(8*sizeof(float3)*videowidth*videoheight*(1.33333))), d.oldthreadnum), core);

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamCreate(d.streams + i);
    }

    data = (ButterData *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < STREAMNUM; i++){
        data->streams[i] = d.streams[i];
    }
    data->maxshared = devattr.sharedMemPerBlock;    

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
}

}