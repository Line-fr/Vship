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
            hipSetDevice(i);
            hipGetDevice(&device);
            hipGetDeviceProperties(&devattr, device);
            ss << "GPU " << i << ": " << devattr.name << std::endl;
        }
    } else {
        hipSetDevice(gpuid);
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);
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

extern "C"{
Vship_Version Vship_GetVersion(){
    Vship_Version res;
    res.major = 4; res.minor = 0; res.minorMinor = 0;
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
        return (Vship_Exception)e.type;
    }
    if (gpu_id >= count){
        return Vship_BadDeviceArgument;
    }
    hipDeviceProp_t devattr;
    hipGetDeviceProperties(&devattr, gpu_id);
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
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len){
    std::string cppstr = errorMessage((VSHIPEXCEPTTYPE)exception);
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    memcpy(out_message, cppstr.c_str(), std::min(len-1, (int)cppstr.size()));
    out_message[len-1] = '\0'; //end character
    return cppstr.size()+1;
}

Vship_Exception Vship_SetDevice(int gpu_id){
    int numgpu;
    Vship_Exception errcount = Vship_GetDeviceCount(&numgpu);
    if (errcount != Vship_NoError){
        return errcount;
    }
    if (gpu_id >= numgpu){
        return Vship_BadDeviceArgument;
    }
    hipSetDevice(gpu_id);
    return Vship_NoError;
}

Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size){
    hipError_t erralloc = hipHostMalloc(ptr, size);
    if (erralloc != hipSuccess){
        return Vship_OutOfRAM;
    }
    return Vship_NoError;
}

Vship_Exception Vship_PinnedFree(void* ptr){
    hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess){
        return Vship_BadPointer;
    }
    return Vship_NoError;
}

RessourceManager<ssimu2::SSIMU2ComputingImplementation*> HandlerManagerSSIMU2;
RessourceManager<butter::ButterComputingImplementation*> HandlerManagerButteraugli;
RessourceManager<cvvdp::CVVDPComputingImplementation*> HandlerManagerCVVDP;

Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerSSIMU2.allocate();
    HandlerManagerSSIMU2.lock.lock();
    HandlerManagerSSIMU2.elements[handler->id] = new ssimu2::SSIMU2ComputingImplementation();
    auto& implem = *HandlerManagerSSIMU2.elements[handler->id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        implem.init(src_colorspace, dis_colorspace);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    if (handler.id >= HandlerManagerSSIMU2.elements.size()){
        HandlerManagerSSIMU2.lock.unlock();
        return Vship_BadHandler;
    }
    auto* implem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    delete implem;
    HandlerManagerSSIMU2.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeSSIMU2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation& ssimu2computingimplem = *HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run(srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerButteraugli.allocate();
    HandlerManagerButteraugli.lock.lock();
    HandlerManagerButteraugli.elements[handler->id] = new butter::ButterComputingImplementation();
    auto* implem = HandlerManagerButteraugli.elements[handler->id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        //Qnorm = 2 by default to mimic old behavior
        implem->init(src_colorspace, dis_colorspace, Qnorm, intensity_multiplier);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    if (handler.id >= HandlerManagerButteraugli.elements.size()){
        HandlerManagerButteraugli.lock.unlock();
        return Vship_BadHandler;
    }
    auto* implem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    delete implem;
    HandlerManagerButteraugli.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeButteraugli(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation* buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<float, float, float> res = buttercomputingimplem->run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
        score->normQ = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_CVVDPInit(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerCVVDP.allocate();
    HandlerManagerCVVDP.lock.lock();
    HandlerManagerCVVDP.elements[handler->id] = new cvvdp::CVVDPComputingImplementation();
    auto* implem = HandlerManagerCVVDP.elements[handler->id];
    HandlerManagerCVVDP.lock.unlock();

    std::string model_key(model_key_cstr);
    try{
        implem->init(src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_CVVDPFree(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    if (handler.id >= HandlerManagerCVVDP.elements.size()){
        HandlerManagerCVVDP.lock.unlock();
        return Vship_BadHandler;
    }
    auto* implem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    try{
        implem->destroy();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    delete implem;
    HandlerManagerCVVDP.free(handler.id);
    return err;
}

Vship_Exception Vship_ResetCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem->flushTemporalRing();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

/*
//this video allows loading images to the temporal filter of CVVDP without computing the whole metric.
//this is useful to start computing at the middle of a video, you can put previous frames with this.
Vship_Exception Vship_LoadCVVDPUint16(Vship_CVVDPHandler handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem.loadImageToRing<UINT16>(srcp1, srcp2, stride, stride2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

//this video allows loading images to the temporal filter of CVVDP without computing the whole metric.
//this is useful to start computing at the middle of a video, you can put previous frames with this.
Vship_Exception Vship_LoadCVVDPFloat(Vship_CVVDPHandler handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        cvvdpcomputingimplem.loadImageToRing<FLOAT>(srcp1, srcp2, stride, stride2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}
*/

Vship_Exception Vship_ComputeCVVDP(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    HandlerManagerCVVDP.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    cvvdp::CVVDPComputingImplementation* cvvdpcomputingimplem = HandlerManagerCVVDP.elements[handler.id];
    HandlerManagerCVVDP.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = cvvdpcomputingimplem->run(dstp, dststride, srcp1, srcp2, lineSize, lineSize2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

} //extern "C"