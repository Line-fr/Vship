#ifndef PREPROCESSHPP
#define PREPROCESSHPP

#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include<vector>
#include <chrono>
#include <thread>
#include<exception>
#include<set>
#include <mutex>
#include <condition_variable>
#include <cassert>


#if defined __CUDACC__
    #include <cuda_fp16.h>
    #define LOWLEVEL
    #define hipMemcpyDtoH(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyDeviceToHost)
    #define hipMemcpyHtoD(x, y, z) cudaMemcpy(x, y, z, cudaMemcpyHostToDevice)
    #define hipMemcpyDtoHAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyDeviceToHost, w)
    #define hipMemcpyHtoDAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyHostToDevice, w)
    #define hipMemcpyDtoDAsync(x, y, z, w) cudaMemcpyAsync(x, y, z, cudaMemcpyDeviceToDevice, w)
    #define hipMemcpyPeer cudaMemcpyPeer
    #define hipMemcpyPeerAsync cudaMemcpyPeerAsync
    #define hipMalloc cudaMalloc
    #define hipFree cudaFree
    #define hipDeviceSynchronize cudaDeviceSynchronize
    #define hipSetDevice cudaSetDevice
    #define hipDeviceProp_t cudaDeviceProp
    #define hipGetDeviceCount cudaGetDeviceCount
    #define hipDeviceptr_t void*
    #define hipGetDevice cudaGetDevice
    #define hipGetDeviceProperties cudaGetDeviceProperties
    #define hipError_t cudaError_t
    #define hipGetErrorString cudaGetErrorString
    #define hipStream_t cudaStream_t
    #define hipStreamAddCallback cudaStreamAddCallback
    #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
    #define hipSuccess cudaSuccess
    #define hipGetLastError cudaGetLastError
    #define hipStreamCreate cudaStreamCreate
    #define hipStreamDestroy cudaStreamDestroy
    #define hipEventCreate cudaEventCreate
    #define hipEventDestroy cudaEventDestroy
    #define hipEventSynchronize cudaEventSynchronize
    #define hipEventRecord cudaEventRecord
    #define hipEvent_t cudaEvent_t
    #define hipEventElapsedTime cudaEventElapsedTime
    #define hipDeviceSetCacheConfig cudaDeviceSetCacheConfig 
    #define hipFuncCachePreferShared cudaFuncCachePreferShared
    #define hipFuncCachePreferNone cudaFuncCachePreferNone
    #define hipFuncCachePreferL1 cudaFuncCachePreferL1
    #define hipFuncCachePreferEqual cudaFuncCachePreferEqual
    #define hipMemGetInfo cudaMemGetInfo
    #define hipMemsetAsync cudaMemsetAsync
    #define hipMemset cudaMemset
    #define hipMallocAsync cudaMallocAsync
    #define hipFreeAsync cudaFreeAsync
    #define hipHostFree cudaFreeHost
    #define hipFreeHost cudaFreeHost
    #define hipHostMalloc cudaMallocHost
    #define hipStreamSynchronize cudaStreamSynchronize
#else
    #include<hip/hip_runtime.h>
    #include<hip/hip_fp16.h>
#endif

#ifdef _WIN32
    #define aligned_alloc(a, b) _aligned_malloc(b, a)
#endif


hipError_t err_hip;

#define GPU_CHECK(x)\
err_hip = (x);\
if (err_hip != hipSuccess)\
{\
   	printf("%s in %s at %d\n", hipGetErrorString(err_hip),  __FILE__, __LINE__);\
}

#define GAUSSIANSIZE 8
#define SIGMA 1.5f
#define PI  3.14159265359
#define TAU 6.28318530718

enum InputMemType {UINT16, HALF, FLOAT};

#endif