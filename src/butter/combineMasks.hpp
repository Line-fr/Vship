namespace butter{

__device__ float MaskY(float delta) {
    const float offset = 0.829591754942f;
    const float scaler = 0.451936922203f;
    const float mul = 2.5485944793f;
    const float c = mul / ((scaler * delta) + offset);
    const float retval = (1.0 + c)*(1.0/(0.79079917404f*17.83f));
    return retval * retval;
}

__device__ float MaskDcY(float delta) {
    const float offset = 0.20025578522f;
    const float scaler = 3.87449418804f;
    const float mul = 0.505054525019f;
    const float c = mul / ((scaler * delta) + offset);
    const float retval = (1.0 + c)*(1.0/(0.79079917404f*17.83f));
    return retval * retval;
}

__launch_bounds__(256)
__global__ void computeDiffmap_Kernel(float* mask, float* block_diff_dc0, float* block_diff_dc1, float* block_diff_dc2, float* block_diff_ac0, float* block_diff_ac1, float* block_diff_ac2, float* dst, int width){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    float val = mask[x];
    float maskval = MaskY(val);
    float maskval_dc = MaskDcY(val);

    dst[x] = sqrtf(maskval * (block_diff_ac0[x] + block_diff_ac1[x] + block_diff_ac2[x]) + maskval_dc * (block_diff_dc0[x] + block_diff_dc1[x] + block_diff_dc2[x]));
    //if ((x == 1841628 && width == 1080*1920) || ((x == 959/2 * 960 + 348/2 && width == 540*960))) printf("final result: %f with ac: %f, %f, %f, dc: %f, %f, %f, mask: %f.\n", dst[x], block_diff_ac0[x], block_diff_ac1[x], block_diff_ac2[x], block_diff_dc0[x], block_diff_dc1[x], block_diff_dc2[x], mask[x]);
}

void computeDiffmap(float* mask, float* block_diff_dc0, float* block_diff_dc1, float* block_diff_dc2, float* block_diff_ac0, float* block_diff_ac1, float* block_diff_ac2, float* dst, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    computeDiffmap_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mask, block_diff_dc0, block_diff_dc1, block_diff_dc2, block_diff_ac0, block_diff_ac1, block_diff_ac2, dst, width);
    GPU_CHECK(hipGetLastError());
}

}