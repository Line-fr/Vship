#pragma once

namespace cvvdp{

__global__ void preGaussianPreCompute_kernel(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[thid], band, channel);
    p1[thid] *= S;
    p2[thid] *= S;
}

void preGaussianPreCompute(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    preGaussianPreCompute_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(Lbkg, p1, p2, width, height, channel, band, csfhandle);
}

//works for a band  but does all channels. It writes result on source ref
//work on 16x16 blocks
__global__ void computeD_Kernel(float* R0, float* R1, float* R2, float* R3, float* T0, float* T1, float* T2, float* T3, const int width, const int height, GaussianHandle gaussianhandle){
    const int64_t x = (threadIdx.x + blockIdx.x*16);
    const int64_t y = (threadIdx.y + blockIdx.y*16);
    const int64_t id = y*width + x;

    float* gaussiankernel = gaussianhandle.gaussiankernel_d;
    float* gaussiankernel_integral = gaussianhandle.gaussiankernel_integral_d;
    __shared__ float sharedmem[1024];

    const float powmaskc = powf(10, mask_c);

    GaussianSmartSharedLoadMin(sharedmem, R0, T0, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm0 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[0]);
    __syncthreads();

    GaussianSmartSharedLoadMin(sharedmem, R1, T1, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm1 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[1]);
    __syncthreads();

    GaussianSmartSharedLoadMin(sharedmem, R2, T2, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm2 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[2]);
    __syncthreads();

    GaussianSmartSharedLoadMin(sharedmem, R3, T3, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm3 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[3]);
    __syncthreads();

    const float Cmask0 = fmaf(blurred_Cm0, powf(xcm_weights[0], 2), fmaf(blurred_Cm1, powf(xcm_weights[1], 2), fmaf(blurred_Cm2, powf(xcm_weights[2], 2), blurred_Cm3*powf(xcm_weights[3], 2))));
    const float Cmask1 = fmaf(blurred_Cm0, powf(xcm_weights[4], 2), fmaf(blurred_Cm1, powf(xcm_weights[5], 2), fmaf(blurred_Cm2, powf(xcm_weights[6], 2), blurred_Cm3*powf(xcm_weights[7], 2))));
    const float Cmask2 = fmaf(blurred_Cm0, powf(xcm_weights[8], 2), fmaf(blurred_Cm1, powf(xcm_weights[9], 2), fmaf(blurred_Cm2, powf(xcm_weights[10], 2), blurred_Cm3*powf(xcm_weights[11], 2))));
    const float Cmask3 = fmaf(blurred_Cm0, powf(xcm_weights[12], 2), fmaf(blurred_Cm1, powf(xcm_weights[13], 2), fmaf(blurred_Cm2, powf(xcm_weights[14], 2), blurred_Cm3*powf(xcm_weights[15], 2))));

    if (id >= width*height) return;
    const float Du0 = powf(abs(R0[id] - T0[id]), mask_p)/(1+Cmask0);
    const float Du1 = powf(abs(R1[id] - T1[id]), mask_p)/(1+Cmask1);
    const float Du2 = powf(abs(R2[id] - T2[id]), mask_p)/(1+Cmask2);
    const float Du3 = powf(abs(R3[id] - T3[id]), mask_p)/(1+Cmask3);

    const float max_v = powf(10, d_max);
    const float D0 = max_v*Du0/(max_v+Du0);
    const float D1 = max_v*Du1/(max_v+Du1);
    const float D2 = max_v*Du2/(max_v+Du2);
    const float D3 = max_v*Du3/(max_v+Du3);

    //we write D inside Reference planes
    R0[id] = D0;
    R1[id] = D1;
    R2[id] = D2;
    R3[id] = D3;

    //if (id == 100) printf("D width %d: %f %f %f %f\n", width, D0, D1, D2, D3);
}

void computeD(float* R0, float* R1, float* R2, float* R3, float* T0, float* T1, float* T2, float* T3, const int width, const int height, GaussianHandle& gaussianhandle, hipStream_t stream){
    int th_x = 16;
    int th_y = 16;
    int bl_x = (width +th_x-1)/th_x;
    int bl_y = (height +th_y-1)/th_y;
    computeD_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(R0, R1, R2, R3, T0, T1, T2, T3, width, height, gaussianhandle);
}

__global__ void computeD_baseband_kernel(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[thid], band, channel);
    p1[thid] = abs(p1[thid] - p2[thid])*S;
}

void computeD_baseband(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    computeD_baseband_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(Lbkg, p1, p2, width, height, channel, band, csfhandle);
}

}