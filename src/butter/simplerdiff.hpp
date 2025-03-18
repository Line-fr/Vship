namespace butter{


__launch_bounds__(256)
__global__ void samenoisediff_Kernel(float* src1, float* dst, int width, float w){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    dst[x] += w * src1[x] * src1[x];
}

void samenoisediff(float* src1, float* dst, int width, float w, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    samenoisediff_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, dst, width, w);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void diffclamp_Kernel(float* src1, float* src2, float* dst, int width, float maxclamp){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    float v0 = abs(src1[x]);
    float v1 = abs(src2[x]);

    if (v0 > maxclamp) v0 = maxclamp;
    if (v1 > maxclamp) v1 = maxclamp;
    dst[x] = v1 - v0;
}

void diffclamp(float* src1, float* src2, float* dst, int width, float maxclamp, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    diffclamp_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width, maxclamp);
    GPU_CHECK(hipGetLastError());
}

__host__ void sameNoiseLevels(Plane_d p1, Plane_d p2, Plane_d output, Plane_d temp1, Plane_d temp2, const float sigma, const float w, const float maxclamp, float* gaussiankernel){
    //output should not be used because we are going to += it that is why we need 2 temporaries
    diffclamp(p1.mem_d, p2.mem_d, temp1.mem_d, p1.width*p1.height, maxclamp, p1.stream);
    temp1.blur(temp2, sigma, 0.0f, gaussiankernel);
    samenoisediff(temp1.mem_d, output.mem_d, p1.width*p1.height, w, p1.stream);
}

__launch_bounds__(256)
__global__ void L2diff_Kernel(float* src1, float* src2, float* dst, int width, float w){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    const float diff = src1[x] - src2[x];
    dst[x] += w * diff * diff;
    //if (x == 10000) printf("l2diff_dc : %f from %f, %f\n", dst[x], src1[x], src2[x]);
}

void L2diff(float* src1, float* src2, float* dst, int width, float w, hipStream_t stream){
    if (w == 0) return;
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    L2diff_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width, w);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void L2AsymDiff_Kernel(float* src1, float* src2, float* dst, int width, float w_0gt1, float w_0lt1){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    const float diff = src1[x] - src2[x];
    dst[x] += w_0gt1 * diff * diff;

    const float fabs0 = abs(src1[x]);
    const float too_small = 0.4 * fabs0;
    const float too_big = fabs0;

    if (src1[x] < 0) {
        if (src2[x] > -too_small) {
            float v = src2[x] + too_small;
            dst[x] += w_0lt1 * v * v;
        } else if (src2[x] < -too_big) {
            float v = -src2[x] - too_big;
            dst[x] += w_0lt1 * v * v;
        }
    } else {
        if (src2[x] < too_small) {
            float v = too_small - src2[x];
            dst[x] += w_0lt1 * v * v;
        } else if (src2[x] > too_big) {
            float v = src2[x] - too_big;
            dst[x] += w_0lt1 * v * v;
        }
    }
}

void L2AsymDiff(float* src1, float* src2, float* dst, int width, float w_0gt1, float w_0lt1, hipStream_t stream){
    if (w_0gt1 == 0.0f && w_0lt1 == 0.0f) return;
    w_0gt1 *= 0.8;
    w_0lt1 *= 0.8;
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    L2AsymDiff_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width, w_0gt1, w_0lt1);
    GPU_CHECK(hipGetLastError());
}

}