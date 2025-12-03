#ifndef EXCEPTIONSHPP
#define EXCEPTIONSHPP

#include<exception>

//to parse the error message automatically: here is the recipe

//VshipException
//{Error Enum Name}: {message related to error type}
// - At line {line} of {file}

enum VSHIPEXCEPTTYPE{
    NoError = 0,

    //vship internal issues
    OutOfVRAM = 1,
    OutOfRAM = 2,
    HIPError = 12,
    
    //input issues
    BadDisplayModel = 3,
    DifferingInputType = 4,
    NonRGBSInput = 5, //should never happen since .resize should give RGBS always
    
    //Device related
    DeviceCountError = 6,
    NoDeviceDetected = 7,
    BadDeviceArgument = 8,
    BadDeviceCode = 9,

    //API related
    BadHandler = 10,
    BadPointer = 11,

    //should not be used
    BadErrorType = 13,
};

std::string errorMessage(VSHIPEXCEPTTYPE type){
    switch (type){
        case NoError:
        return "NoError: When everything is right, you should not be searching for problems but appreciate what was given to you.";

        case OutOfVRAM:
        return "OutOfVRAM: Vship was not able to perform GPU memory allocation. (Advice) Reduce or Set numStream argument";

        case OutOfRAM:
        return "OutOfRAM: Vship was not able to allocate CPU memory. This is a rare error that should be reported. Check your RAM usage";
        
        case HIPError:
        return "InternalError: A GPU Call failed inside Vship. This may be due to a bad environment but is likely due to a bug in Vship.";

        case BadDisplayModel:
        return "BadDisplayModel: Vship was not able to find a corresponding model as specified. (Advice) See valid models in the doc or remove this option";

        case DifferingInputType:
        return "DifferingInputType: Vship received 2 videos with different properties. (Advice) verify that they have the same width, height and length";

        case NonRGBSInput:
        return "NonRGBSInput: Vship did not manage to get RGBS format of your inputs. This should not happen. (Advice) try converting yourself to RGBS";
    
        case DeviceCountError:
        return "DeviceCountError: Vship was unable to verify the number of GPU on your system. (Advice) Did you select the correct binary for your device AMD/NVIDIA. (Advice) if linux AMD, are you in video and render groups?";
        
        case NoDeviceDetected:
        return "NoDeviceDetected: Vship found no device on your system. (Advice) Did you select the correct binary for your device AMD/NVIDIA. (Advice) if linux AMD, are you in video and render groups?";
        
        case BadDeviceArgument:
        return "BadDeviceArgument: Vship received a bad gpu_id argument either you specified a number >= to your gpu count, either it was negative";

        case BadDeviceCode:
        return "BadDeviceCode: Vship was unable to run a simple GPU Kernel. This usually indicate that the code was compiled for the wrong architecture. (Advice) Try to compile vship yourself, eventually replace --offload-arch=native to your arch";

        case BadHandler:
        return "BadHandler: The handler used was not allocated. This indicates an issue in Vship API Usage. Did you call the Init function or altered the handler id?";

        case BadPointer:
        return "BadPointer: The pointer passed as an argument was not valid and returned an error. (Advice) If this happened while trying to free, was it allocated using the same method and not modified?";

        case BadErrorType:
        return "BadErrorType: There was an unknown error";
    }
    return "BadErrorType: There was an unknown error"; //this will not happen but the compiler will be happy
}

class VshipError : public std::exception
{
    std::string file;
    int line;
    std::string detail = "";
public:
    VSHIPEXCEPTTYPE type;
    VshipError(VSHIPEXCEPTTYPE type, const std::string filename, const int line, const std::string detail = "") : std::exception(), type(type), file(filename), line(line), detail(detail){
    }
    
    std::string getErrorMessage() const
    {
        std::stringstream ss;
        ss << "VshipException" << std::endl;
        ss << errorMessage(type) << std::endl;
        ss << " - At line " << line << " of " << file << std::endl;
        if (detail != "") ss << "Detail: " << detail << std::endl;
        return ss.str();
    }
};

#define GPU_CHECK(x)\
{hipError_t err_hip = x;\
if (err_hip != hipSuccess) throw VshipError(HIPError, __FILE__, __LINE__, hipGetErrorString(err_hip));}

#endif