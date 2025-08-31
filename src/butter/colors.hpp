namespace butter{

__device__ float gamma(float v) {
    return fmaf(19.245013259874995f, logf(v + 9.9710635769299145), -23.16046239805755f);
}

__device__ inline void rgb_to_linrgbfunc(float& a){
    if (a < 0){
        a = -powf(-a, 2.4);
    } else {
        a = powf(a, 2.4);
    }
}

__global__ void linearrgb_kernel(float* src1, float* src2, float* src3, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    rgb_to_linrgbfunc(src1[x]);
    rgb_to_linrgbfunc(src2[x]);
    rgb_to_linrgbfunc(src3[x]);

    //if ((x == 376098 && width*height == 1080*1920)) printf("After Linear %f, %f, %f\n", src1[x], src2[x], src3[x]);
}

__device__ inline void butterOpsinAbsorbance(float3& a, bool clamp = false){
    float3 out;
    out.x = fmaf(0.29956550340058319f, a.x,
    fmaf(0.63373087833825936f, a.y,
    fmaf(0.077705617820981968f, a.z,
    1.7557483643287353f)));

    out.y = fmaf(0.22158691104574774f, a.x,
    fmaf(0.69391388044116142f, a.y,
    fmaf(0.0987313588422f, a.z,
    1.7557483643287353f)));

    out.z = fmaf(0.02f, a.x,
    fmaf(0.02f, a.y,
    fmaf(0.20480129041026129f, a.z,
    12.226454707163354f)));

    if (clamp){
        out.x = max(out.x, 1.7557483643287353f);
        out.y = max(out.y, 1.7557483643287353f);
        out.z = max(out.z, 12.226454707163354f);
    }

    a = out;
}

__global__ void opsinDynamicsImage_kernel(float* src1, float* src2, float* src3, float* blurred1, float* blurred2, float* blurred3, int64_t width, int64_t height, float intensity_multiplier){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;

    float3 sensitivity;
    float3 src = {src1[x], src2[x], src3[x]};
    float3 blurred = {blurred1[x], blurred2[x], blurred3[x]};
    //float3 oldsrc = src; float3 oldblurred = blurred;
    src *= intensity_multiplier;
    blurred *= intensity_multiplier;
    butterOpsinAbsorbance(blurred, true);
    blurred = max(blurred, 1e-4f);
    sensitivity.x = gamma(blurred.x) / blurred.x;
    sensitivity.y = gamma(blurred.y) / blurred.y;
    sensitivity.z = gamma(blurred.z) / blurred.z;
    sensitivity = max(sensitivity, 1e-4f);
    butterOpsinAbsorbance(src, false);
    src *= sensitivity;
    src.x = max(src.x, 1.7557483643287353f);
    src.y = max(src.y, 1.7557483643287353f);
    src.z = max(src.z, 12.226454707163354f);

    //make positive + export

    src1[x] = src.x - src.y; 
    src2[x] = src.x + src.y; 
    src3[x] = src.z;
    //if ((x == 376098 && width*height == 1080*1920)) printf("%f, %f, %f and %f, %f, %f to %f, %f, %f with %f, %f, %f sens\n", oldsrc.x, oldsrc.y, oldsrc.z, oldblurred.x, oldblurred.y, oldblurred.z, src1[x], src2[x], src3[x], sensitivity.x, sensitivity.y, sensitivity.z);
}

void opsinDynamicsImage(float* src[3], float* temp[3], int64_t width, int64_t height, GaussianHandle& gaussianHandle, float intensity_multiplier, hipStream_t stream){
    //change src from SRGB to opsin dynamic XYB
    int64_t th_x = std::min((int64_t)256, width*height);
    int64_t bl_x = (width*height-1)/th_x + 1;
    //printf("initial adress: %llu\n", (unsigned long long)src[0].mem_d);
    for (int i = 0; i < 3; i++){
        blurDstNoTemp(temp[i], src[i], width, height, gaussianHandle, 0, stream);
    }
    opsinDynamicsImage_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src[0], src[1], src[2], temp[0], temp[1], temp[2], width, height, intensity_multiplier);
    GPU_CHECK(hipGetLastError());
}

void linearRGB(float* src[3], int64_t width, int64_t height, hipStream_t stream){
    int64_t th_x = std::min((int64_t)256, width*height);
    int64_t bl_x = (width*height-1)/th_x + 1;
    linearrgb_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src[0], src[1], src[2], width, height);
}

}