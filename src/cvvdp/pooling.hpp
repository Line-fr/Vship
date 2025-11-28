#pragma once

namespace cvvdp{

//pointer jumping, start with 1024 threads
template<int power, bool applyPower, bool applyInversePower>
__global__ void reduceSum(float* dst, float* src, int64_t size, bool divide){
    const int64_t global_thid = threadIdx.x + blockIdx.x * blockDim.x;
    const int local_thid = threadIdx.x;
    constexpr int threadnum = 1024;

    __shared__ float pointerJumpingBuffer[threadnum]; //one float per thread

    if (global_thid >= size){
        pointerJumpingBuffer[local_thid] = 0;
    } else {
        if constexpr (applyPower) {
            pointerJumpingBuffer[local_thid] = powf(src[global_thid], power);
            //if (!applyInversePower && global_thid == 13*1024 + 64 && power == 2 && size == 1184264) printf("%lld .. %f\n", size, src[global_thid]);
        } else {
            pointerJumpingBuffer[local_thid] = src[global_thid];
        }
    }

    __syncthreads();

    int next = 1;
    while (next < threadnum){
        //if (!applyInversePower && size == 1184264 && applyPower && power == 2 && global_thid == 13*1024 + 64) printf("width: %lld, next: %d, val: %f\n", size, next, pointerJumpingBuffer[local_thid]);
        if (local_thid + next < threadnum && (local_thid%(next*2) == 0)){
            pointerJumpingBuffer[local_thid] += pointerJumpingBuffer[local_thid+next];
        }
        next *= 2;
        __syncthreads();
    }

    if (local_thid == 0){
        float res = pointerJumpingBuffer[0];
        //if (!applyInversePower && power == 2 && global_thid == 0) printf("We got0 %f at size %lld\n", res, size);
        if (divide) res /= size;
        //if (applyInversePower && power == 2 && global_thid == 0) printf("We got1 %f\n", res);
        if constexpr (applyInversePower) {
            res = powf(res, 1.f/(float)power);
        }
        //if (!applyPower && !applyInversePower && power == 2 && global_thid == 0) printf("We got2 %f\n", res);
        dst[blockIdx.x] = res;
    }
}

//the result will be at temp[0]. We suppose that temp is of the same size as src
template<int power>
void computeMean(float* src, float* temp, int64_t size, bool divide, hipStream_t stream){
    constexpr int th_x = 1024;
    int bl_x;

    float* final_dst = temp; //to contain temp[0]
    float* tempbuffer[3] = {temp+1, temp+1+(size+1023)/1024, src};
    int oscillator = 2; //corresponds to current source except the first time
    while (size > 1024){
        bl_x = (size+th_x-1)/th_x;
        int destination = (oscillator == 2) ? 0 : (oscillator^1);
        if (oscillator == 2){
            reduceSum<power, true, false><<<dim3(bl_x), dim3(th_x), 0, stream>>>(tempbuffer[destination], tempbuffer[oscillator], size, divide);
        } else {
            reduceSum<power, false, false><<<dim3(bl_x), dim3(th_x), 0, stream>>>(tempbuffer[destination], tempbuffer[oscillator], size, false);
        }
        GPU_CHECK(hipGetLastError());
        oscillator = destination;
        size = (size+1023)/1024;
    }
    bl_x = 1;
    if (oscillator == 2){
        reduceSum<power, true, true><<<dim3(bl_x), dim3(th_x), 0, stream>>>(final_dst, tempbuffer[oscillator], size, divide);
    } else {
        reduceSum<power, false, true><<<dim3(bl_x), dim3(th_x), 0, stream>>>(final_dst, tempbuffer[oscillator], size, false);
    }
    GPU_CHECK(hipGetLastError());
}

}