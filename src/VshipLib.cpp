#include "VshipColor.h"
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "butter/vapoursynth.hpp"
#include "ssimu2/vapoursynth.hpp"
#include "cvvdp/vapoursynth.hpp"
#include "util/gpuhelper.hpp"

static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::stringstream ss;
    int count, device;
    hipDeviceProp_t devattr;

    //we don't need a full check at that point
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }
    
    if (count <= gpuid || gpuid < 0){
        vsapi->mapSetError(out, VshipError(BadDeviceArgument, __FILE__, __LINE__).getErrorMessage().c_str());
        return;
    }

    if (error != peSuccess){
        //no gpu_id was selected
        for (int i = 0; i < count; i++){
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipGetDevice(&device));
            GPU_CHECK(hipGetDeviceProperties(&devattr, device));
            ss << "GPU " << i << ": " << devattr.name << std::endl;
        }
    } else {
        GPU_CHECK(hipSetDevice(gpuid));
        GPU_CHECK(hipGetDevice(&device));
        GPU_CHECK(hipGetDeviceProperties(&devattr, device));
        ss << "Name: " << devattr.name << std::endl;
        ss << "MultiProcessorCount: " << devattr.multiProcessorCount << std::endl;
        //ss << "ClockRate: " << ((float)devattr.clockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda 13
        ss << "MaxSharedMemoryPerBlock: " << devattr.sharedMemPerBlock << " bytes" << std::endl;
        ss << "WarpSize: " << devattr.warpSize << std::endl;
        ss << "VRAMCapacity: " << ((float)devattr.totalGlobalMem)/1000000000 << " GB" << std::endl;
        ss << "MemoryBusWidth: " << devattr.memoryBusWidth << " bits" << std::endl;
        //ss << "MemoryClockRate: " << ((float)devattr.memoryClockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda13
        ss << "Integrated: " << devattr.integrated << std::endl;
        ss << "PassKernelCheck : " << (int)helper::gpuKernelCheck() << std::endl;
    }
    vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(4, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;qnorm:int:opt;intensity_multiplier:float:opt;distmap:int:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("CVVDP", "reference:vnode;distorted:vnode;model_name:data:opt;resizeToDisplay:int:opt;distmap:int:opt;gpu_id:int:opt;", "clip:vnode;", cvvdp::CVVDPCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}

//let's define the API
#define EXPORTVSHIPLIB //to use dllexport for windows
#include "VshipAPI.h"

Vship_Exception convertLocalErrorToAPI(const VshipError& e){
    Vship_Exception res;
    res.type = (Vship_ExceptionType)e.type;
    res.line = e.line;
    int msgLen = std::min(e.file.size(), (size_t)255);
    memcpy(res.file, e.file.c_str(), msgLen);
    res.file[msgLen] = '\0';
    msgLen = std::min(e.detail.size(), (size_t)255);
    memcpy(res.details, e.detail.c_str(), msgLen);
    return res;
}

VshipError convertAPIErrorToLocal(const Vship_Exception& e){
    return VshipError((VSHIPEXCEPTTYPE)e.type, std::string(e.file), e.line, std::string(e.details));
}

Vship_Exception APINoError(){
    Vship_Exception res;
    res.type = Vship_NoError;
    res.file[0] = '\0';
    res.details[0] = '\0';
    res.line = 0;
    return res;
}

extern "C"{
Vship_Version Vship_GetVersion(){
    Vship_Version res;
    res.major = 4; res.minor = 0; res.minorMinor = 1;
    #if defined __CUDACC__
    res.backend = Vship_Cuda;
    #else
    res.backend = Vship_HIP;
    #endif
    return res;
}

Vship_Exception Vship_GetDeviceCount(int* number){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        return convertLocalErrorToAPI(e);
    }
    *number = count;
    return APINoError();
}

Vship_Exception Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        return convertLocalErrorToAPI(e);
    }
    if (gpu_id >= count){
        return convertLocalErrorToAPI(VshipError(BadDeviceArgument, __FILE__, __LINE__));
    }
    hipDeviceProp_t devattr;
    GPU_CHECK(hipGetDeviceProperties(&devattr, gpu_id));
    memcpy(device_info->name, devattr.name, 256); //256 char to copy
    device_info->VRAMSize = devattr.totalGlobalMem;
    device_info->integrated = devattr.integrated;
    device_info->MultiProcessorCount = devattr.multiProcessorCount;
    device_info->WarpSize = devattr.warpSize;
    return APINoError();
}

Vship_Exception Vship_GPUFullCheck(int gpu_id){
    try{
        helper::gpuFullCheck(gpu_id);
    } catch (const VshipError& e){
        return convertLocalErrorToAPI(e);
    }
    return APINoError();
}

int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len){
    std::string cppstr = convertAPIErrorToLocal(exception).getErrorMessage();
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    memcpy(out_message, cppstr.c_str(), std::min(len-1, (int)cppstr.size()));
    out_message[len-1] = '\0'; //end character
    return cppstr.size()+1;
}

Vship_Exception Vship_SetDevice(int gpu_id){
    int numgpu;
    Vship_Exception errcount = Vship_GetDeviceCount(&numgpu);
    if (errcount.type != Vship_NoError){
        return errcount;
    }
    if (gpu_id >= numgpu){
        return convertLocalErrorToAPI(VshipError(BadDeviceArgument, __FILE__, __LINE__));
    }
    GPU_CHECK(hipSetDevice(gpu_id));
    return APINoError();
}

Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size){
    hipError_t erralloc = hipHostMalloc(ptr, size);
    if (erralloc != hipSuccess){
        return convertLocalErrorToAPI(VshipError(OutOfRAM, __FILE__, __LINE__));
    }
    return APINoError();
}

Vship_Exception Vship_PinnedFree(void* ptr){
    hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess){
        return convertLocalErrorToAPI(VshipError(BadPointer, __FILE__, __LINE__));
    }
    return APINoError();
}

RessourceManager<ssimu2::SSIMU2ComputingImplementation*> HandlerManagerSSIMU2;
RessourceManager<butter::ButterComputingImplementation*> HandlerManagerButteraugli;
RessourceManager<cvvdp::CVVDPComputingImplementation*> HandlerManagerCVVDP;

Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace){
    Vship_Exception err = APINoError();
    handler->id = HandlerManagerSSIMU2.allocate();
    HandlerManagerSSIMU2.lock.lock();
    HandlerManagerSSIMU2.elements[handler->id] = new ssimu2::SSIMU2ComputingImplementation();
    auto& implem = *HandlerManagerSSIMU2.elements[handler->id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        implem.init(src_colorspace, dis_colorspace);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return APINoError();
}

Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler){
    Vship_Exception err = APINoError();
    HandlerManagerSSIMU2.lock.lock();
    if (handler.id >= HandlerManagerSSIMU2.elements.size()){
        HandlerManagerSSIMU2.lock.unlock();
        return convertLocalErrorToAPI(VshipError(BadHandler, __FILE__, __LINE__));
    }
    auto* implem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    delete implem;
    HandlerManagerSSIMU2.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeSSIMU2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = APINoError();
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation& ssimu2computingimplem = *HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run(srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier){
    Vship_Exception err = APINoError();
    handler->id = HandlerManagerButteraugli.allocate();
    HandlerManagerButteraugli.lock.lock();
    HandlerManagerButteraugli.elements[handler->id] = new butter::ButterComputingImplementation();
    auto* implem = HandlerManagerButteraugli.elements[handler->id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        //Qnorm = 2 by default to mimic old behavior
        implem->init(src_colorspace, dis_colorspace, Qnorm, intensity_multiplier);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler){
    Vship_Exception err = APINoError();
    HandlerManagerButteraugli.lock.lock();
    if (handler.id >= HandlerManagerButteraugli.elements.size()){
        HandlerManagerButteraugli.lock.unlock();
        return convertLocalErrorToAPI(VshipError(BadHandler, __FILE__, __LINE__));
    }
    auto* implem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    delete implem;
    HandlerManagerButteraugli.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeButteraugli(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = APINoError();
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation* buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<double, double, double> res = buttercomputingimplem->run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
        score->normQ = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

Vship_Exception Vship_CVVDPInit(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr){
    Vship_Exception err = APINoError();
    handler->id = HandlerManagerCVVDP.allocate();
    HandlerManagerCVVDP.lock.lock();
    HandlerManagerCVVDP.elements[handler->id] = new cvvdp::CVVDPComputingImplementation();
    auto* implem = HandlerManagerCVVDP.elements[handler->id];
    HandlerManagerCVVDP.lock.unlock();

    std::string model_key(model_key_cstr);
    try{
        implem->init(src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

Vship_Exception Vship_CVVDPFree(Vship_CVVDPHandler handler){
    Vship_Exception err = APINoError();
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        return convertLocalErrorToAPI(VshipError(BadHandler, __FILE__, __LINE__));
    }
    auto* implem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    delete implem;
    HandlerManagerCVVDP.free(handler.id);
    return err;
}

Vship_Exception Vship_ResetCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = APINoError();
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem->flushTemporalRing();
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

//The temporalFilter stays loaded but the score goes back to neutral
Vship_Exception Vship_ResetScoreCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = APINoError();
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem->flushOnlyScore();
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

//this function allows loading images to the temporal filter of CVVDP without computing metric.
//this is useful to start computing at the middle of a video, you can put previous frames with this.
Vship_Exception Vship_LoadTemporalCVVDP(Vship_CVVDPHandler handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = APINoError();
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem->loadImageToRing(srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

Vship_Exception Vship_ComputeCVVDP(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = APINoError();
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = cvvdpcomputingimplem->run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        err = convertLocalErrorToAPI(e);
    }
    return err;
}

} //extern "C"