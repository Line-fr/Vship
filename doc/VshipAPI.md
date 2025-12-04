# Introduction

Vship API has been made to allow low level access to Vship interal metric without having to use FFVship or vapoursynth. It typically allow to retrieve SSIMU2 scores, butteraugli distortion map. It also benefits from not requiring complex compilation processes by being a simple shared library.

# Compilation and Usage

## How to install VshipAPI

VshipAPI is included when installing Vship the vapoursynth plugin. The API itself is contained within vship.dll which also contains the vapoursynth API support. As such, installing VshipAPI only requires doing `make build` (or `make buildcuda` for nvidia GPUs) and `make install`. For windows, there is no standard lib installation folder. As such, the compiled .dll which can also be found on the release page and [VshipAPI Header](../src/VshipAPI.h) need to be installed manually at a known place for use and compilation of your software.

## Usage of VshipAPI inside the code

Including the header and using VshipAPI function is all that is required. As a .h header, it is directly usable in C or C++ but can also be ported to work in other languages.

Example:

```Cpp
#include "VshipAPI.h

int main(){
    int number_of_gpu;
    Vship_Exception err = Vship_GetDeviceCount(&number_of_gpu);
    return 0;
}
```

Later in the document there will be more concrete examples and detailed description of each functions

## Compilation of a software using VshipAPI

As stated before, you only need the header file and the shared library.
As such compilation is as simple as 

`gcc mycode.c -I HeaderDirectory -L LibraryDirectory -lvship`

Note that the -I and -L arguments are optional depending on the placement of the header and library. 
For the resulting executable to work, it will need to be able to find libvship.so. On Ubuntu, it might be necessary to add the library path (/usr/local/lib) to LD_LIBRARY_PATH environment variable. Another option is to put libvship.so/dll next to your executable.

To verify that everything works for you, you can try compiling yourself [this example](../test/apitest.c)

# API detailed tutorial and description

## Interactive Tutorial

For this tutorial, we will suppose that we wish to retrieve a Butteraugli distortion map using Vship and that you possess the YUV420 8bits BT709 source and distorted for which we will compute this distmap. Vship can accept a lot of difference input colorspaces as can be seen in [here](../src/VshipColor.h)

### Error Managment

For this tutorial, we will not bother too much with errors with will create a simple preprocessor wrapper to handle vship errors:

```Cpp
#include<stdio.h>
#include<stdlib.h>

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetDetailedLastError(errmsg, 1024);\
    printf("Vship Error occured: %s", errmsg);\
    exit(1);\
}
```

This macro will simply print vship error message if an error was to occur

### Verify that vship will work before using it

```Cpp
//check if gpu 0 exist and if it can work with vship
ErrorCheck(Vship_GPUFullCheck(0));
```

Thanks to our macro, the error message will be displayed to the user before exiting, and nothing will happen if it goes well.

### Defining our Colorspace

```Cpp
//we supposed the same for source and distorted but they can have different colorspaces
//here we define the classic Limited BT709 YUV420 source
Vship_Colorspace_t colorspace;
colorspace.width = image_width;
colorspace.height = image_height;
colorspace.sample = Vship_SampleUINT8;
colorspace.range = Vship_RangeLimited;
colorspace.subsampling = {1, 1};
colorspace.colorFamily = Vship_ColorYUV;
colorspace.YUVMatrix = Vship_MATRIX_BT709;
colorspace.transferFunction = Vship_TRC_BT709;
colorspace.primaries = Vship_PRIMARIES_BT709;
```

### Create a Butteraugli Handler

Vship being made for maximal throughput, it does some preprocessing. A handler may be used for processing a lot of frames (but only one at a time). To create a handler, you can do this

```Cpp
Vship_ButteraugliHandler butterhandler;
//we put the colorspace twice, one for source and one for distorted
//This will initialize a handler with intensity_target 203 in butteraugli
ErrorCheck(Vship_ButteraugliInit(&butterhandler, colorspace, colorspace, 203.));
```

### Frame Processing

Our frame must be planar and of type: `const uint8*[3]` but the actual data inside the plane will be of the type specified in the colorspace. We also need to specify the stride for each plane

```Cpp
//I suppose that you have already set the data of the frame
const uint8* image1[3];
const uint8* image2[3];
const int64_t strides1[3];
const int64_t strides2[3];

//the result will be here!
//let's allocate the space required to store the distortion map. It is always float
//you can use image_width as a stride in this case
const uint8* distortionMap = (uint8_t*)malloc(sizeof(float)*image_height*image_stride);

Vship_ButteraugliScore score;

//ask vship to compute!
//it will handle color conversion and metric computing
ErrorCheck(Vship_ComputeButteraugliUint16(butterhandler, &score, distortionMap, image_stride, image1, image2, strides1, strides2));

//now score contains butteraugli scores and distortionMap contains the distortion map!
```

### Cleanup

We want to avoid leaks so there is a little step to add

```Cpp
//free the distortion map we just allocated
free(distortionMap);

//free the butteraugli handler
ErrorCheck(Vship_ButteraugliFree(butterhandler));
```

## Good practices

### Performance concerns

It is important if you wish to maximize throughput to create about 3 handlers that process frames in parallel. You can test the importance of this by playing with the `-g` option of FFVship. 

Note that it can increase VRAM usage. You can verify the amount of VRAM used per working handler [here (SSIMU2)](SSIMULACRA2.md) and [here (Butteraugli)](BUTTERAUGLI.md). You can also retrieve the total amount of VRAM of the GPU using `Vship_GetDeviceInfo`.

### Video Metrics

CVVDP is a Video Metric, as such, a fully score can only be gotten by processing every single frames through a single handler in the right order. 

## Details on every function

### Vship_GetVersion

This function returns a Vship_Version type containing the version of Vship
For example, at the time I am writing this documentation we have

```Cpp
Vship_Version v = Vship_GetVersion();
v.major; //4
v.minor; //0
v.minorMinor; //0
// => 4.0.0
```

It is also possible to retrieve the type of GPU present (NVIDIA or HIP) using v.backend

### Vship_GetDeviceCount(int* number)

This function returns an exception if it failed to get the wanted information. In case of success, the integer passed as an argument will be equal to the number of GPU detected by Vship.

### Vship_SetDevice(int gpu_id)

This function allows to choose the detected GPU to run vship on. If the gpu_id chosen is not valid, an exception is returned. You can have information to choose the GPU you wish to use using the following function. But in general, GPU 0 is the most relevant GPU to use.

### Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id)

This function allows to retrieve some GPU information in the Vship_DeviceInfo struct defined here:

```Cpp
struct{
    char name[256];
    //size in bytes
    uint64_t VRAMSize;
    //iGPU? (boolean)
    int integrated;
    int MultiProcessorCount;
    int WarpSize;
}
```

You choose the GPU wiht gpu_id and device_info will possess the wanted informations.

### Vship_GPUFullCheck(int gpu_id)

This function allows to check if Vship will be able to run on the given GPU or not. It can return a variety of exceptions depending on the check that fails. If no error is returned, the main concerns for later parts could be a failed RAM or VRAM allocation. This function is highly recommended as done in the interactive example

### int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len)

This function is used to get a detailed error message from an exception. It is useful to manage errors lazifully while still giving advices to solve the issue. There are 2 ways to use it:

```Cpp
char errmsg[1024];
//will not overflow, the function is aware of the size of the allocation "1024". However, if a message was to be bigger thana 1024 characters, it would be cut.
Vship_GetErrorMessage(error, errmsg, 1024);

//----------------
//the other way to not overflow and to be sure to retrieve the whole message
char* errmsg2;
int predicted_size = Vship_GetErrorMessage(error, NULL, 0); //retrieve size
//allocate
errmsg2 = (char*)malloc(sizeof(char)*predicted_size);
//retrieve the message
Vship_GetErrorMessage(error, errmsg2, predicted_size);

//do what you want with the errmsg and then free
free(errmsg2);
```

### int Vship_GetDetailedLastError(char* out_message, int len);

This function works similarly to Vship_GetErrorMessage. But the error message that is returned contains more details. This function will hold onto the last error that happened in vship as a whole.This means that if a call has an error and another right after does not, this function will still return the error message of the error.

Note that The error is stored as thread_local which means only the thread getting the error can retrieve the details but also that it is threadsafe

### Vship_PinnedMalloc(void** ptr, uint64_t size)

This function allows a special and **expensive** type of allocation. The allocated data can be sent to the GPU with less latency and using less memory bandwidth by eliminating a memory copy. As such, for optimal speed, you should be using these planes for the images you plan on sending to the gpu for compute but only if you were to reuse the memory allocated because allocating and freeing for each frame with this function will occur slowdown.

```Cpp
uint8_t* mypointer;
VshipException err = Vship_PinnedMalloc(&mypointer, allocationsize);
//now you can use mypointer. But don't forget to free using the next function! it cannot be freed with just free(mypointer);
```

### Vship_PinnedFree(void* ptr)

This function is used to free the pinned memory allocated above.

```Cpp
uint8_t* mypointer;
Vship_PinnedMalloc(&mypointer, 1);
mypointer[0] = 42;
//yeay we wrote something
Vship_PinnedFree(mypointer); //cleanup using this function
```

### Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace)

This function is used to perform some preprocessing using colorspaces. It creates a handler to be used on the compute function. A handler should only be used to process one frame at a time, should not be used after free but it can process multiple frames sequentially. It is possible and even recommended to create multiple Handler to process multiple frames in parallel.

### Vship_SSIMU2Free(Vship_SSIMU2Handler handler)

To avoid leaks, every handler that was allocated should be freed later using this function.

### Vship_ComputeSSIMU2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize1[3], const int64_t lineSize2[3]);

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames who each have their own strides (bytes per line) per plane. The color conversion is done by vship with respect to the colorspace given at init to the handler.

### Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier)

This function is used to perform some preprocessing using colorspaces. It creates a handler to be used on the compute function. A handler should only be used to process one frame at a time, should not be used after free but it can process multiple frames sequentially. It is possible and even recommended to create multiple Handler to process multiple frames in parallel.

The intensity multiplier corresponds to the screen luminosity in nits. It is usually set at 203 nits or 80 nits.
Qnorm allows getting an arbitrary norm of the Butteraugli distortion map in the ButteruagliScore object. By default you can set it to 2.

### Vship_ButteraugliFree(Vship_ButteraugliHandler handler)

To avoid leaks, every handler that was allocated should be freed later using this function.

### Vship_ComputeButteraugli(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride)

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames who each have their own strides (bytes per line) per plane. The color conversion is done by vship with respect to the colorspace given at init to the handler.

It is possible to retrieve the distortion map of butteraugli. If you supply NULL to dstp, the distortion map will be discarded without even going back to the CPU. As such, only the score will be obtained. However, if you supply a valid pointer allocated of the right size `sizeof(float)*image_height*dststride`, the distortion will be retrieved and stored here. image_height here represent the new height of the image. If the colorspace specifies a resize and a crop, you will need to take that into account.

### Vship_CVVDPInit(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr)

This function is used to perform some preprocessing using colorspaces. It creates a handler to be used on the compute function. A handler should only be used to process one frame at a time, should not be used after free but it can process multiple frames sequentially. It is not recommended to create multiple Handler to process multiple frames in parallel since you need to give the handler a coherent temporal influx of frames.

This handler will have its own temporal filter. As such, you need to feed frames in the right order to this handler.

the video fps value is important since it controls the size of the temporal filter and can increase or lower VRAM value.

for more information about resizeToDisplay and models, you can refer to [this page](CVVDP.md)

### Vship_CVVDPFree(Vship_CVVDPHandler handler);

To avoid leaks, every handler that was allocated should be freed later using this function.

### Vship_ResetCVVDP(Vship_CVVDPHandler handler);

If you wish to reuse a handler but for a different video or even a different part of the video, you may wish to reset the temporal filter without recreating a whole new handler (which can take more time). This is what this function is made for. It resets the temporall filter of a given handler.

### Vship_ComputeCVVDP(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3])

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames who each have their own strides (bytes per line) per plane. The color conversion is done by vship with respect to the colorspace given at init to the handler.

It is possible to retrieve the distortion map of CVVDP. If you supply NULL to dstp, the distortion map will be discarded without even going back to the CPU. As such, only the score will be obtained. However, if you supply a valid pointer allocated of the right size `sizeof(float)*image_height*dststride`, the distortion will be retrieved and stored here. image_height here represent the new height of the image. If the colorspace specifies a resize and a crop, you will need to take that into account.