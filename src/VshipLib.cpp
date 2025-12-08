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
    vspapi->registerFunction("CVVDP", "reference:vnode;distorted:vnode;model_name:data:opt;model_config_json:data:opt;resizeToDisplay:int:opt;distmap:int:opt;gpu_id:int:opt;", "clip:vnode;", cvvdp::CVVDPCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}

//let's define the API
#define EXPORTVSHIPLIB //to use dllexport for windows
#include "VshipAPI.h"

template<typename ImplementationType>
struct HandlerData{
    ImplementationType implem;
};

RessourceManager<HandlerData<ssimu2::SSIMU2ComputingImplementation>*> HandlerManagerSSIMU2;
RessourceManager<HandlerData<butter::ButterComputingImplementation>*> HandlerManagerButteraugli;
RessourceManager<HandlerData<cvvdp::CVVDPComputingImplementation>*> HandlerManagerCVVDP;

thread_local VshipError lastError = VshipError(NoError, "Your Heart", 0, "");

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
        lastError = e;
        return (Vship_Exception)e.type;
    }
    *number = count;
    return Vship_NoError;
}

Vship_Exception Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    if (gpu_id >= count){
        lastError = VshipError(BadDeviceArgument, __FILE__, __LINE__, "while using Vship_GetDeviceInfo");
        return Vship_BadDeviceArgument;
    }
    hipDeviceProp_t devattr;
    try{
        GPU_CHECK(hipGetDeviceProperties(&devattr, gpu_id));
    } catch (VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    memcpy(device_info->name, devattr.name, 256); //256 char to copy
    device_info->VRAMSize = devattr.totalGlobalMem;
    device_info->integrated = devattr.integrated;
    device_info->MultiProcessorCount = devattr.multiProcessorCount;
    device_info->WarpSize = devattr.warpSize;
    return Vship_NoError;
}

Vship_Exception Vship_GPUFullCheck(int gpu_id){
    try{
        helper::gpuFullCheck(gpu_id);
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len){
    std::string cppstr = errorMessage((VSHIPEXCEPTTYPE)exception);
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    const int msglength = std::min(len-1, (int)cppstr.size());
    memcpy(out_message, cppstr.c_str(), msglength);
    out_message[msglength] = '\0'; //end character
    return cppstr.size()+1;
}

int Vship_GetDetailedLastError(char* out_message, int len){
    std::string cppstr = lastError.getErrorMessage();
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    const int msglength = std::min(len-1, (int)cppstr.size());
    memcpy(out_message, cppstr.c_str(), msglength);
    out_message[msglength] = '\0'; //end character
    return cppstr.size()+1;
}

Vship_Exception Vship_SetDevice(int gpu_id){
    int numgpu;
    Vship_Exception errcount = Vship_GetDeviceCount(&numgpu);
    if (errcount != Vship_NoError){
        return errcount;
    }
    if (gpu_id >= numgpu){
        lastError = VshipError(BadDeviceArgument, __FILE__, __LINE__, "In Vship_SetDevice");
        return Vship_BadDeviceArgument;
    }
    try{
        GPU_CHECK(hipSetDevice(gpu_id));
    } catch (VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size){
    hipError_t erralloc = hipHostMalloc(ptr, size);
    if (erralloc != hipSuccess){
        lastError = VshipError(OutOfRAM, __FILE__, __LINE__, "In Vship_PinnedMalloc");
        return Vship_OutOfRAM;
    }
    return Vship_NoError;
}

Vship_Exception Vship_PinnedFree(void* ptr){
    hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess){
        lastError = VshipError(BadPointer, __FILE__, __LINE__, "Failed to call hipHostFree on pointer given to Vship_PinnedFree");
        return Vship_BadPointer;
    }
    return Vship_NoError;
}

Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerSSIMU2.allocate();
    HandlerManagerSSIMU2.lock.lock();
    HandlerManagerSSIMU2.elements[handler->id] = new HandlerData<ssimu2::SSIMU2ComputingImplementation>();
    auto* handlerdata = HandlerManagerSSIMU2.elements[handler->id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        handlerdata->implem.init(src_colorspace, dis_colorspace);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    if (handler.id >= HandlerManagerSSIMU2.elements.size()){
        HandlerManagerSSIMU2.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_SSIMU2Free is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    auto* handlerdata = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        handlerdata->implem.destroy();
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    delete handlerdata;
    HandlerManagerSSIMU2.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeSSIMU2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    if (handler.id >= HandlerManagerSSIMU2.elements.size()){
        HandlerManagerSSIMU2.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeSSIMU2 is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = handlerdata->implem.run(srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerButteraugli.allocate();
    HandlerManagerButteraugli.lock.lock();
    HandlerManagerButteraugli.elements[handler->id] = new HandlerData<butter::ButterComputingImplementation>();
    auto* handlerdata = HandlerManagerButteraugli.elements[handler->id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        //Qnorm = 2 by default to mimic old behavior
        handlerdata->implem.init(src_colorspace, dis_colorspace, Qnorm, intensity_multiplier);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    if (handler.id >= HandlerManagerButteraugli.elements.size()){
        HandlerManagerButteraugli.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ButteraugliFree is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    auto* handlerdata = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        handlerdata->implem.destroy();
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    delete handlerdata;
    HandlerManagerButteraugli.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeButteraugli(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    if (handler.id >= HandlerManagerButteraugli.elements.size()){
        HandlerManagerButteraugli.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeButteraugli is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<double, double, double> res = handlerdata->implem.run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
        score->normQ = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_CVVDPInit2(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr, const char* model_config_json_cstr){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerCVVDP.allocate();
    HandlerManagerCVVDP.lock.lock();
    HandlerManagerCVVDP.elements[handler->id] = new HandlerData<cvvdp::CVVDPComputingImplementation>();
    auto* handlerdata = HandlerManagerCVVDP.elements[handler->id];
    HandlerManagerCVVDP.lock.unlock();

    std::string model_key(model_key_cstr);
    std::string model_config_json = "";
    if (model_config_json_cstr != NULL){
        model_config_json = std::string(model_config_json_cstr);
    } 
    try{
        handlerdata->implem.init(src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key, model_config_json);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_CVVDPInit(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr){
    return Vship_CVVDPInit2(handler, src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key_cstr, NULL);
}

Vship_Exception Vship_CVVDPFree(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_CVVDPFree is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    auto* handlerdata = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    try{
        handlerdata->implem.destroy();
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    delete handlerdata;
    HandlerManagerCVVDP.free(handler.id);
    return err;
}

Vship_Exception Vship_ResetCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ResetCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        handlerdata->implem.flushTemporalRing();
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

//The temporalFilter stays loaded but the score goes back to neutral
Vship_Exception Vship_ResetScoreCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ResetScoreCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        handlerdata->implem.flushOnlyScore();
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

//this function allows loading images to the temporal filter of CVVDP without computing metric.
//this is useful to start computing at the middle of a video, you can put previous frames with this.
Vship_Exception Vship_LoadTemporalCVVDP(Vship_CVVDPHandler handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_LoadTemporalCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        handlerdata->implem.loadImageToRing(srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeCVVDP(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = handlerdata->implem.run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

} //extern "C"