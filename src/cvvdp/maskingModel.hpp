#pragma once

namespace cvvdp{

__global__ void preGaussianPreCompute_kernel(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float ch_gain = (channel == 1) ? 1.45f : 1.f;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[thid], band, channel);
    p1[thid] *= S * ch_gain;
    p2[thid] *= S * ch_gain;
    //if (thid == 0) printf("preGaussian ends with %f %f using sensi %f\n", p1[thid], p2[thid], S);
}

void preGaussianPreCompute(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    preGaussianPreCompute_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(Lbkg, p1, p2, width, height, channel, band, csfhandle);
    GPU_CHECK(hipGetLastError());
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

    GaussianSmartSharedLoadMinAbs(sharedmem, R0, T0, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm0 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[0]);
    __syncthreads();

    GaussianSmartSharedLoadMinAbs(sharedmem, R1, T1, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm1 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[1]);
    __syncthreads();

    GaussianSmartSharedLoadMinAbs(sharedmem, R2, T2, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm2 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[2]);
    __syncthreads();

    GaussianSmartSharedLoadMinAbs(sharedmem, R3, T3, x, y, width, height);
    GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm3 = powf(powmaskc*abs(sharedmem[(threadIdx.y+8)*32+threadIdx.x+8]), mask_q[3]);
    __syncthreads();

    const float Cmask0 = fmaf(blurred_Cm0, powf(2.f, xcm_weights[0]), fmaf(blurred_Cm1, powf(2.f, xcm_weights[4]), fmaf(blurred_Cm2, powf(2.f, xcm_weights[8]), blurred_Cm3*powf(2.f, xcm_weights[12]))));
    const float Cmask1 = fmaf(blurred_Cm0, powf(2.f, xcm_weights[1]), fmaf(blurred_Cm1, powf(2.f, xcm_weights[5]), fmaf(blurred_Cm2, powf(2.f, xcm_weights[9]), blurred_Cm3*powf(2.f, xcm_weights[13]))));
    const float Cmask2 = fmaf(blurred_Cm0, powf(2.f, xcm_weights[2]), fmaf(blurred_Cm1, powf(2.f, xcm_weights[6]), fmaf(blurred_Cm2, powf(2.f, xcm_weights[10]), blurred_Cm3*powf(2.f, xcm_weights[14]))));
    const float Cmask3 = fmaf(blurred_Cm0, powf(2.f, xcm_weights[3]), fmaf(blurred_Cm1, powf(2.f, xcm_weights[7]), fmaf(blurred_Cm2, powf(2.f, xcm_weights[11]), blurred_Cm3*powf(2.f, xcm_weights[15]))));
    //if (id == 0) printf("Cmask: %f %f %f %f from %f %f %f %f\n", Cmask0, Cmask1, Cmask2, Cmask3, blurred_Cm0, blurred_Cm1, blurred_Cm2, blurred_Cm3);

    if (x >= width || y >= height) return;
    const float Du0 = powf(abs(R0[id] - T0[id]), mask_p)/(1+Cmask0);
    const float Du1 = powf(abs(R1[id] - T1[id]), mask_p)/(1+Cmask1);
    const float Du2 = powf(abs(R2[id] - T2[id]), mask_p)/(1+Cmask2);
    const float Du3 = powf(abs(R3[id] - T3[id]), mask_p)/(1+Cmask3);

    //if (id == 0) printf("D width %d: %f %f %f %f vs %f %f %f %f with mask %f %f %f %f\n", width, R0[id], R1[id], R2[id], R3[id], T0[id], T1[id], T2[id], T3[id], Cmask0, Cmask1, Cmask2, Cmask3);
    //if (id == 13*1024 + 64 && width == 827) printf("D width %d: %f %f %f %f\n", width, Du0, Du1, Du2, Du3);

    const float max_v = powf(10, d_max);
    const float D0 = max_v*Du0/(max_v+Du0);
    const float D1 = max_v*Du1/(max_v+Du1);
    const float D2 = max_v*Du2/(max_v+Du2);
    const float D3 = max_v*Du3/(max_v+Du3);

    //we write D inside Reference planes along with the channel multiplier
    R0[id] = D0 * 1.f;
    R1[id] = D1 * ch_chrom_w;
    R2[id] = D2 * ch_chrom_w;
    R3[id] = D3 * ch_trans_w;

    //if (id == 13*1024 + 64 && width == 827) printf("D id/width %lld/%d: %f %f %f %f\n", id, width, D0, D1, D2, D3);
}

void computeD(float* R0, float* R1, float* R2, float* R3, float* T0, float* T1, float* T2, float* T3, const int width, const int height, GaussianHandle& gaussianhandle, hipStream_t stream){
    int th_x = 16;
    int th_y = 16;
    int bl_x = (width +th_x-1)/th_x;
    int bl_y = (height +th_y-1)/th_y;
    computeD_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(R0, R1, R2, R3, T0, T1, T2, T3, width, height, gaussianhandle);
    GPU_CHECK(hipGetLastError());
}

__global__ void computeD_baseband_kernel(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[0], band, channel);
    //if (thid == 0) printf("ComputeDbasedband D: %f, T_f %f R_f %f S %f Lbkg %f\n", abs(p1[thid] - p2[thid])*S, p2[thid], p1[thid], S, Lbkg[0]);
    p1[thid] = abs(p1[thid] - p2[thid])*S * baseband_weight[channel]; //the weight is put here to avoid another kernel later doing that
}

void computeD_baseband(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    computeD_baseband_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(Lbkg, p1, p2, width, height, channel, band, csfhandle);
    GPU_CHECK(hipGetLastError());
}

}