#ifndef VSHIP_API_HEADER
#define VSHIP_API_HEADER

#include <stdint.h>

#if defined(_WIN32)
#if defined(EXPORTVSHIPLIB)
#define EXPORTPREPROCESS __declspec(dllexport)
#else
#define EXPORTPREPROCESS __declspec(dllimport)
#endif
#else
#define EXPORTPREPROCESS
#endif

#ifdef __cplusplus
extern "C"{
#endif

typedef enum{
    Vship_HIP = 0,
    Vship_Cuda = 1,
} Vship_Backend;

typedef struct {
    int major;
    int minor;
    int minorMinor;
    Vship_Backend backend;
} Vship_Version;

EXPORTPREPROCESS Vship_Version Vship_GetVersion();

//this is general purpose, it contains error that cannot be encountered using the API
typedef enum{
    Vship_NoError = 0,

    //vship internal issues
    Vship_OutOfVRAM,
    Vship_OutOfRAM,
    
    //input issues
    Vship_BadDisplayModel,
    Vship_DifferingInputType,
    Vship_NonRGBSInput, //should never happen since .resize should give RGBS always
    
    //Device related
    Vship_DeviceCountError,
    Vship_NoDeviceDetected,
    Vship_BadDeviceArgument,
    Vship_BadDeviceCode,

    //API related
    Vship_BadHandler,
    Vship_BadPointer,

    //should not be used
    Vship_BadErrorType,
} Vship_Exception;

//Get the number of GPU
EXPORTPREPROCESS Vship_Exception Vship_GetDeviceCount(int* number);

//default device is default by HIP/Cuda (0)
EXPORTPREPROCESS Vship_Exception Vship_SetDevice(int gpu_id);

//more features might be added later on in this struct
typedef struct{
    char name[256];
    uint64_t VRAMSize; //size in bytes
    int integrated; //iGPU? (boolean)
    int MultiProcessorCount;
    int WarpSize;
} Vship_DeviceInfo;

EXPORTPREPROCESS Vship_Exception Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id);

//very useful function allowing to see if vship is going to work there are multiple errors possible returned
EXPORTPREPROCESS Vship_Exception Vship_GPUFullCheck(int gpu_id);

//you can allocate typically 1024 bytes to retrieve the full error message
//however, if you want the exact amount, the integer returned is the size needed. So you can use len=0 to retrieve the size, allocate and then the correct len.
EXPORTPREPROCESS int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len);

//for maximum throughput, it is recommend to use 3 SSIMU2Handler with each a thread to use in parallel
//is only an id to refer to an object in an array in the API dll.
//this is because the original object contains types that are not represented without hip and the original code.
typedef struct{
    int id;
} Vship_SSIMU2Handler;

EXPORTPREPROCESS Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size);

EXPORTPREPROCESS Vship_Exception Vship_PinnedFree(void* ptr);

//handler pointer will be replaced, it is a return value. Don't forget to free it after usage.
EXPORTPREPROCESS Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, int width, int height);

//handler pointer can be discarded after this function.
EXPORTPREPROCESS Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler);

//the frame is not overwritten
//when input is RGB with BT709 transfer function float frame
EXPORTPREPROCESS Vship_Exception Vship_ComputeSSIMU2Float(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

EXPORTPREPROCESS Vship_Exception Vship_ComputeSSIMU2Floatv2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);

//the frame is not overwritten
//when input is RGB with BT709 transfer function uint16_t frame
EXPORTPREPROCESS Vship_Exception Vship_ComputeSSIMU2Uint16(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

EXPORTPREPROCESS Vship_Exception Vship_ComputeSSIMU2Uint16v2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);

typedef struct{
    int id;
} Vship_ButteraugliHandler;

typedef struct{
    int norm2;
    int norm3;
    int norminf;
} Vship_ButteraugliScore;

//handler pointer will be replaced, it is a return value. Don't forget to free it after usage.
EXPORTPREPROCESS Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, int width, int height, float intensity_multiplier);

//this version allows specifying a Qnorm instead of getting norm2
EXPORTPREPROCESS Vship_Exception Vship_ButteraugliInitv2(Vship_ButteraugliHandler* handler, int width, int height, int Qnorm, float intensity_multiplier);

//handler pointer can be discarded after this function.
EXPORTPREPROCESS Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler);

//the frame is not overwritten
//dstp must either be NULL (in this case, the distortion map will never be retrieved from the gpu)
//or be allocated of size dststride*height
//when input is RGB with BT709 transfer function float frame
//output in score
EXPORTPREPROCESS Vship_Exception Vship_ComputeButteraugliFloat(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

EXPORTPREPROCESS Vship_Exception Vship_ComputeButteraugliFloatv2(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);

//the frame is not overwritten
//dstp must either be NULL (in this case, the distortion map will never be retrieved from the gpu)
//or be allocated of size dststride*height
//when input is RGB with BT709 transfer function uint16_t frame
//output in score
EXPORTPREPROCESS Vship_Exception Vship_ComputeButteraugliUint16(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

EXPORTPREPROCESS Vship_Exception Vship_ComputeButteraugliUint16v2(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);

typedef struct{
    int id;
} Vship_CVVDPHandler;

//handler pointer will be replaced, it is a return value. Don't forget to free it after usage.
EXPORTPREPROCESS Vship_Exception Vship_CVVDPInit(Vship_CVVDPHandler* handler, int width, int height, float fps, const char* model_key_cstr);

//handler pointer can be discarded after this function.
EXPORTPREPROCESS Vship_Exception Vship_CVVDPFree(Vship_CVVDPHandler handler);

//the frame is not overwritten
//dstp must either be NULL (in this case, the distortion map will never be retrieved from the gpu)
//or be allocated of size dststride*height
//when input is RGB with BT709 transfer function uint16_t frame
//output in score
EXPORTPREPROCESS Vship_Exception Vship_ComputeCVVDPUint16(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);

//the frame is not overwritten
//dstp must either be NULL (in this case, the distortion map will never be retrieved from the gpu)
//or be allocated of size dststride*height
//when input is RGB with BT709 transfer function float frame
//output in score
EXPORTPREPROCESS Vship_Exception Vship_ComputeCVVDPFloat(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2);


#ifdef __cplusplus
} //extern "C"
#endif
#endif //ifndef VSHIP_API_HEADER