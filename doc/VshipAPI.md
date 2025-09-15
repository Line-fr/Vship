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

`gcc mycode.c -I HeaderDirectory -L LibraryDirectory -l:vship.so`

Note that the -I and -L arguments are optional depending on the placement of the header and library. 
For the resulting executable to work, it will need to be able to find vship.so. On Ubuntu, it might be necessary to add the library path (/usr/local/lib) to LD_LIBRARY_PATH environment variable. Another option is to put vship.so/dll next to your executable.

To verify that everything works for you, you can try compiling yourself [this example](../test/apitest.c)

# API detailed tutorial and description

## Interactive Tutorial

For this tutorial, we will suppose that we wish to retrieve a Butteraugli distortion map using Vship and that you possess the planar RGB with BT709 transfer frames for which we will compute this distmap.

### Error Managment

For this tutorial, we will not bother too much with errors with will create a simple preprocessor wrapper to handle vship errors:

```Cpp
#include<stdio.h>
#include<stdlib.h>

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetErrorMessage(err, errmsg, 1024);\
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

### Create a Butteraugli Handler

Vship being made for maximal throughput, it does some preprocessing. A handler may be used for processing a lot of frames (but only one at a time). To create a handler, you can do this

```Cpp
Vship_ButteraugliHandler butterhandler;
//This will initialize a handler with intensity_target 203 in butteraugli
ErrorCheck(Vship_ButteraugliInit(&butterhandler, imaage_width, image_height, 203.));
```

### Frame Processing

Our RGB planar BT709 transfer frames must be of type: `const uint8*[3]` but the actual data inside the plane will be in uint16_t in this example

```Cpp
//I suppose that you have already set the data of the frame
const uint8* image1[3];
const uint8* image2[3];

//the result will be here!
//let's allocate the space required to store the distortion map. It is always float
//you can use image_width as a stride in this case
const uint8* distortionMap = (uint8_t*)malloc(sizeof(float)*image_height*image_stride);

Vship_ButteraugliScore score;

//ask vship to compute! We use the Uint16 version since our frames are coded as uint16_t
ErrorCheck(Vship_ComputeButteraugliUint16(butterhandler, &score, distortionMap, image_stride, image1, image2, image_stride));

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

### RGB with BT709 transfer?

You might be wondering how to convert your frame to this planar RGB with BT709 transfer. A way to do this is to use [Zimg](https://github.com/sekrit-twc/zimg) which is what is used in FFVship.

### Performance concerns

It is important if you wish to maximize throughput to create about 3 handlers that process frames in parallel. You can test the importance of this by playing with the `-g` option of FFVship. 

Note that it can increase VRAM usage. You can verify the amount of VRAM used per working handler [here (SSIMU2)](SSIMULACRA2.md) and [here (Butteraugli)](BUTTERAUGLI.md). You can also retrieve the total amount of VRAM of the GPU using `Vship_GetDeviceInfo`.

## Details on every function

### Vship_GetVersion

This function returns a Vship_Version type containing the version of Vship
For example, at the time I am writing this documentation we have

```Cpp
Vship_Version v = Vship_GetVersion();
v.major; //3
v.minor; //1
v.minorMinor; //0
// => 3.1.0
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

### Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, int width, int height)

This function is used to perform some preprocessing using the width and height. It creates a handler to be used on the compute function. A handler should only be used to process one frame at a time, should not be used after free but it can process multiple frames sequentially. It is possible and even recommended to create multiple Handler to process multiple frames in parallel.

### Vship_SSIMU2Free(Vship_SSIMU2Handler handler)

To avoid leaks, every handler that was allocated should be freed later using this function.

### Vship_ComputeSSIMU2Float(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames which both have a given stride.

these frames must be planar RGB with BT709 transfer and their color data encoded in 32bit Float.

### Vship_ComputeSSIMU2Uint16(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride);

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames which both have a given stride.

these frames must be planar RGB with BT709 transfer and their color data encoded in 16bit Unsigned integer.

### Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, int width, int height, float intensity_multiplier)

This function is used to perform some preprocessing using the width and height. It creates a handler to be used on the compute function. A handler should only be used to process one frame at a time, should not be used after free but it can process multiple frames sequentially. It is possible and even recommended to create multiple Handler to process multiple frames in parallel.

The intensity multiplier corresponds to the screen luminosity in nits. It is usually set at 203 nits or 80 nits.

### Vship_ButteraugliFree(Vship_ButteraugliHandler handler)

To avoid leaks, every handler that was allocated should be freed later using this function.

### Vship_ComputeButteraugliFloat(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride)

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames which both have a given stride.

these frames must be planar RGB with BT709 transfer and their color data encoded in 32bit Float.

It is possible to retrieve the distortion map of butteraugli. If you supply NULL to dstp, the distortion map will be discarded without even going back to the CPU. As such, only the score will be obtained. However, if you supply a valid pointer allocated of the right size `sizeof(float)*image_height*dststride`, the distortion will be retrieved and stored here.

### Vship_ComputeButteraugliUint16(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride)

Using an allocated handler, you can retrieve the score between two `const uint8_t*[3]` frames which both have a given stride.

these frames must be planar RGB with BT709 transfer and their color data encoded in 16bit Unsigned Integer.

It is possible to retrieve the distortion map of butteraugli. If you supply NULL to dstp, the distortion map will be discarded without even going back to the CPU. As such, only the score will be obtained. However, if you supply a valid pointer allocated of the right size `sizeof(float)*image_height*dststride`, the distortion will be retrieved and stored here.