namespace butter{

//maltaUnit functions are copy pasted from google butteraugli
__device__ float MaltaUnitLF(float* d, const int xs) {
    const int xs3 = 3 * xs;
    float retval = 0.0f;
    {
        // x grows, y constant
        float sum =
            d[-4] +
            d[-2] +
            d[0] +
            d[2] +
            d[4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float sum =
            d[-xs3 - xs] +
            d[-xs - xs] +
            d[0] +
            d[xs + xs] +
            d[xs3 + xs];
        retval += sum * sum;
    }
    {
        // both grow
        float sum =
            d[-xs3 - 3] +
            d[-xs - xs - 2] +
            d[0] +
            d[xs + xs + 2] +
            d[xs3 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float sum =
            d[-xs3 + 3] +
            d[-xs - xs + 2] +
            d[0] +
            d[xs + xs - 2] +
            d[xs3 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float sum =
            d[-xs3 - xs + 1] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 + xs - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float sum =
            d[-xs3 - xs - 1] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + xs + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float sum =
            d[-4 - xs] +
            d[-2 - xs] +
            d[0] +
            d[2 + xs] +
            d[4 + xs];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float sum =
            d[-4 + xs] +
            d[-2 + xs] +
            d[0] +
            d[2 - xs] +
            d[4 - xs];
        retval += sum * sum;
    }
    {
        /* 0_________
        1__*______
        2___*_____
        3_________
        4____0____
        5_________
        6_____*___
        7______*__
        8_________ */
        float sum =
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1______*__
        2_____*___
        3_________
        4____0____
        5_________
        6___*_____
        7__*______
        8_________ */
        float sum =
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 - 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_*_______
        3__*______
        4____0____
        5______*__
        6_______*_
        7_________
        8_________ */
        float sum =
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[0] +
            d[xs + 2] +
            d[xs + xs + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_______*_
        3______*__
        4____0____
        5__*______
        6_*_______
        7_________
        8_________ */
        float sum =
            d[-xs - xs + 3] +
            d[-xs + 2] +
            d[0] +
            d[xs - 2] +
            d[xs + xs - 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2________*
        3______*__
        4____0____
        5__*______
        6*________
        7_________
        8_________ */

        float sum =
            d[xs + xs - 4] +
            d[xs - 2] +
            d[0] +
            d[-xs + 2] +
            d[-xs - xs + 4];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2*________
        3__*______
        4____0____
        5______*__
        6________*
        7_________
        8_________ */
        float sum =
            d[-xs - xs - 4] +
            d[-xs - 2] +
            d[0] +
            d[xs + 2] +
            d[xs + xs + 4];
        retval += sum * sum;
    }
    {
        /* 0__*______
        1_________
        2___*_____
        3_________
        4____0____
        5_________
        6_____*___
        7_________
        8______*__ */
        float sum =
            d[-xs3 - xs - 2] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + xs + 2];
        retval += sum * sum;
    }
    {
        /* 0______*__
        1_________
        2_____*___
        3_________
        4____0____
        5_________
        6___*_____
        7_________
        8__*______ */
        float sum =
            d[-xs3 - xs + 2] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 + xs - 2];
        retval += sum * sum;
    }
    return retval;
}

__device__ float MaltaUnit(float* d, const int xs) {
    const int xs3 = 3 * xs;
    float retval = 0.0f;
    {
        // x grows, y constant
        float sum =
            d[-4] +
            d[-3] +
            d[-2] +
            d[-1] +
            d[0] +
            d[1] +
            d[2] +
            d[3] +
            d[4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float sum =
            d[-xs3 - xs] +
            d[-xs3] +
            d[-xs - xs] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs] +
            d[xs3] +
            d[xs3 + xs];
        retval += sum * sum;
    }
    {
        // both grow
        float sum =
            d[-xs3 - 3] +
            d[-xs - xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + xs + 2] +
            d[xs3 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float sum =
            d[-xs3 + 3] +
            d[-xs - xs + 2] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs + xs - 2] +
            d[xs3 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float sum =
            d[-xs3 - xs + 1] +
            d[-xs3 + 1] +
            d[-xs - xs + 1] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs - 1] +
            d[xs3 - 1] +
            d[xs3 + xs - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float sum =
            d[-xs3 - xs - 1] +
            d[-xs3 - 1] +
            d[-xs - xs - 1] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs + 1] +
            d[xs3 + 1] +
            d[xs3 + xs + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float sum =
            d[-4 - xs] +
            d[-3 - xs] +
            d[-2 - xs] +
            d[-1] +
            d[0] +
            d[1] +
            d[2 + xs] +
            d[3 + xs] +
            d[4 + xs];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float sum =
            d[-4 + xs] +
            d[-3 + xs] +
            d[-2 + xs] +
            d[-1] +
            d[0] +
            d[1] +
            d[2 - xs] +
            d[3 - xs] +
            d[4 - xs];
        retval += sum * sum;
    }
    {
        /* 0_________
        1__*______
        2___*_____
        3___*_____
        4____0____
        5_____*___
        6_____*___
        7______*__
        8_________ */
        float sum =
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + xs + 1] +
            d[xs3 + 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1______*__
        2_____*___
        3_____*___
        4____0____
        5___*_____
        6___*_____
        7__*______
        8_________ */
        float sum =
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs + xs - 1] +
            d[xs3 - 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_*_______
        3__**_____
        4____0____
        5_____**__
        6_______*_
        7_________
        8_________ */
        float sum =
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + 2] +
            d[xs + xs + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_______*_
        3_____**__
        4____0____
        5__**_____
        6_*_______
        7_________
        8_________ */
        float sum =
            d[-xs - xs + 3] +
            d[-xs + 2] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs - 2] +
            d[xs + xs - 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_________
        3______**_
        4____0*___
        5__**_____
        6**_______
        7_________
        8_________ */

        float sum =
            d[xs + xs - 4] +
            d[xs + xs - 3] +
            d[xs - 2] +
            d[xs - 1] +
            d[0] +
            d[1] +
            d[-xs + 2] +
            d[-xs + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2**_______
        3__**_____
        4____0*___
        5______**_
        6_________
        7_________
        8_________ */
        float sum =
            d[-xs - xs - 4] +
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[1] +
            d[xs + 2] +
            d[xs + 3];
        retval += sum * sum;
    }
    {
        /* 0__*______
        1__*______
        2___*_____
        3___*_____
        4____0____
        5____*____
        6_____*___
        7_____*___
        8_________ */
        float sum =
            d[-xs3 - xs - 2] +
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[-xs - 1] +
            d[0] +
            d[xs] +
            d[xs + xs + 1] +
            d[xs3 + 1];
        retval += sum * sum;
    }
    {
        /* 0______*__
        1______*__
        2_____*___
        3_____*___
        4____0____
        5____*____
        6___*_____
        7___*_____
        8_________ */
        float sum =
            d[-xs3 - xs + 2] +
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[-xs + 1] +
            d[0] +
            d[xs] +
            d[xs + xs - 1] +
            d[xs3 - 1];
        retval += sum * sum;
    }
    return retval;
}

__global__ void MaltaDiffMap_Kernel(const float* lum0, const float* lum1, float* block_diff_ac, const int width, const int height, const float w_0gt1, const float w_0lt1, const float norm1, const float len, const float mulli) {
    //each block must be 16*16
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float w_pre0gt1 = mulli * sqrtf(kWeight0 * w_0gt1) / len2;
    const float w_pre0lt1 = mulli * sqrtf(kWeight1 * w_0lt1) / len2;
    const float norm2_0gt1 = w_pre0gt1 * norm1;
    const float norm2_0lt1 = w_pre0lt1 * norm1;

    //this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
    __shared__ float diffs[24*24]; //4 around each border. 24*24 = 1.5² * 16*16 -> around 2.25 values per thread. Note that we can pad in advance with negligible register cost
    //shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)
    int topleftx = blockIdx.x*blockDim.x - 4; int toplefty = blockIdx.y*blockDim.y - 4;

    const int serialind = threadIdx.x + threadIdx.y * blockDim.x;
    const int serialstride = blockDim.x*blockDim.y;
    for (int i = serialind; i < 24*24; i += serialstride){
        int workx = topleftx + i%24; int worky = toplefty + i/24;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = 0.0f;
            continue;
        }

        const float absval = 0.5f * abs(lum0[worky*width + workx]) + 0.5f * abs(lum1[worky*width + workx]);
        const float diff = lum0[worky*width + workx] - lum1[worky*width + workx];
        const float scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
        const float fabs0 = abs(lum0[worky*width + workx]);

        // Secondary half-open quadratic objectives.
        const float too_small = 0.55 * fabs0;
        const float too_big = 1.05 * fabs0;

        float impact = 0.0f;

        if (lum0[worky*width + workx] < 0.0f){
            if (lum1[worky*width + workx] > -too_small){
                impact = lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] < -too_big){
                impact = -lum1[worky*width + workx] - too_big;
            }
        } else {
            if (lum1[worky*width + workx] < too_small){
                impact = -lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] > too_big){
                impact = lum1[worky*width + workx] - too_big;
            }
        }
        impact *= scaler2;

        if (diff < 0.0f){
            diffs[i] -= impact;
        } else {
            diffs[i] += impact;
        }
    }
    __syncthreads(); //diffs is loaded

    //printf("local diffs : %f at %d\n", diffs[(threadIdx.y+4)*24 + threadIdx.x+4], (threadIdx.y+4)*24 + threadIdx.x+4);

    float result = MaltaUnit(diffs+(threadIdx.y+4)*24 + threadIdx.x+4, 24);
    if (x < width && y < height){
        block_diff_ac[y*width + x] += result;
    }
}

__global__ void MaltaDiffMapLF_Kernel(const float* lum0, const float* lum1, float* block_diff_ac, const int width, const int height, const float w_0gt1, const float w_0lt1, const float norm1, const float len, const float mulli) {
    //each block must be 16*16
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float w_pre0gt1 = mulli * sqrtf(kWeight0 * w_0gt1) / len2;
    const float w_pre0lt1 = mulli * sqrtf(kWeight1 * w_0lt1) / len2;
    const float norm2_0gt1 = w_pre0gt1 * norm1;
    const float norm2_0lt1 = w_pre0lt1 * norm1;

    //this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
    __shared__ float diffs[24*24]; //4 around each border. 24*24 = 1.5² * 16*16 -> around 2.25 values per thread. Note that we can pad in advance with negligible register cost
    //shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)
    int topleftx = blockIdx.x*blockDim.x - 4; int toplefty = blockIdx.y*blockDim.y - 4;

    const int serialind = threadIdx.x + threadIdx.y * blockDim.x;
    const int serialstride = blockDim.x*blockDim.y;
    for (int i = serialind; i < 24*24; i += serialstride){
        int workx = topleftx + i%24; int worky = toplefty + i/24;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = 0.0f;
            continue;
        }

        const float absval = 0.5f * abs(lum0[worky*width + workx]) + 0.5f * abs(lum1[worky*width + workx]);
        const float diff = lum0[worky*width + workx] - lum1[worky*width + workx];
        const float scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float scaler2 = norm2_0lt1 / (static_cast<float>(norm1) + absval);
        const float fabs0 = abs(lum0[worky*width + workx]);

        // Secondary half-open quadratic objectives.
        const float too_small = 0.55 * fabs0;
        const float too_big = 1.05 * fabs0;

        float impact = 0.0f;

        if (lum0[worky*width + workx] < 0.0f){
            if (lum1[worky*width + workx] > -too_small){
                impact = lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] < -too_big){
                impact = -lum1[worky*width + workx] - too_big;
            }
        } else {
            if (lum1[worky*width + workx] < too_small){
                impact = -lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] > too_big){
                impact = lum1[worky*width + workx] - too_big;
            }
        }
        impact *= scaler2;

        if (diff < 0.0f){
            diffs[i] -= impact;
        } else {
            diffs[i] += impact;
        }
    }
    __syncthreads(); //diffs is loaded

    float result = MaltaUnitLF(diffs+(threadIdx.y+4)*24 + threadIdx.x+4, 24);
    if (x < width && y < height){
        block_diff_ac[y*width + x] += result;
    }
}

__host__ void MaltaDiffMap(const float* lum0, const float* lum1, float* block_diff_ac, const int width, const int height, const float w_0gt1, const float w_0lt1, const float norm1, hipStream_t stream){
    const float len = 3.75f;
    const float mulli = 0.39905817637f;

    const int th_x = 16;
    const int th_y = 16;
    const int bl_x = (width - 1)/th_x + 1;
    const int bl_y = (height - 1)/th_y + 1;
    MaltaDiffMap_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y)>>>(lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
}

__host__ void MaltaDiffMapLF(const float* lum0, const float* lum1, float* block_diff_ac, const int width, const int height, const float w_0gt1, const float w_0lt1, const float norm1, hipStream_t stream){
    const float len = 3.75f;
    const float mulli = 0.611612573796f;

    const int th_x = 16;
    const int th_y = 16;
    const int bl_x = (width - 1)/th_x + 1;
    const int bl_y = (height - 1)/th_y + 1;
    MaltaDiffMapLF_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
}

}