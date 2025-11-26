namespace butter{

    __device__ inline void XybLowFreqToVals(float& x, float& y, float& b) {
        const float xmuli = 33.832837186260f;
        const float ymuli = 14.458268100570f;
        const float bmuli = 49.87984651440f;
        const float y_to_b_muli = -0.362267051518f;
        b = b + y_to_b_muli * y;
        b = b * bmuli;
        x = x * xmuli;
        y = y * ymuli;
    }
    
    __device__ inline void SuppressHfInBrightAreas(float& hf, float brightness, float mul, float reg) {
        float scaler = mul * reg / (reg + brightness);
        hf *= scaler;
    }

    __device__ inline void SuppressUhfInBrightAreas(float& hf, float brightness, float mul, float reg) {
        float scaler = mul * reg / (reg + brightness);
        hf *= scaler;
    }

    __device__ inline void MaximumClamp(float& v, float maxval) {
        const float kMul = 0.688059627878f;
        if (v >= maxval) {
            v -= maxval;
            v *= kMul;
            v += maxval;
        } else if (v < -maxval) {
            v += maxval;
            v *= kMul;
            v -= maxval;
        }
    }

    __device__ inline void supressXbyY(float& x, float yval, float yw){
        const float s = 0.653020556257f;
        const float scaler = s + (yw * (1.0f - s)) / (yw + yval * yval);
        x *= scaler;
    }

    // Make area around zero more important (2x it until the limit).
    __device__ inline void AmplifyRangeAroundZero(float& x, float w) {
        x = x > w ? x + w : x < -w ? x - w : 2.0f * x;
    }

    // Make area around zero less important (remove it).
    __device__ inline void RemoveRangeAroundZero(float& x, float w) {
        x = x > w ? x - w : x < -w ? x + w : 0.0f;
    }

    __global__ void subarray_removerangearound0_Kernel(float* first, float* second, int64_t width, float w){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        second[x] -= first[x];
        RemoveRangeAroundZero(first[x], w);
    }

    void subarray_removerangearound0(float* first, float* second, int64_t width, float w, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        subarray_removerangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void removerangearound0_Kernel(float* first, int64_t width, float w){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        RemoveRangeAroundZero(first[x], w);
    }

    void removerangearound0(float* first, int64_t width, float w, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        removerangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void subarray_amplifyrangearound0_Kernel(float* first, float* second, int64_t width, float w){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        second[x] -= first[x];
        AmplifyRangeAroundZero(first[x], w);
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void subarray_amplifyrangearound0(float* first, float* second, int64_t width, float w, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        subarray_amplifyrangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void supressXbyY_Kernel(float* first, float* second, int64_t width, float yw){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        supressXbyY(first[x], second[x], yw);
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void supressXbyY(float* first, float* second, int64_t width, float yw, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        supressXbyY_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, yw);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void separateHf_Uhf_Kernel(float* hf, float* uhf, int64_t width){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        const float kMaxclampHf = 28.4691806922f;
        const float kMaxclampUhf = 5.19175294647f;

        MaximumClamp(hf[x], kMaxclampHf);
        uhf[x] -= hf[x];
        MaximumClamp(uhf[x], kMaxclampUhf);
        uhf[x] *= 2.69313763794f;
        hf[x] *= 2.155f;
        AmplifyRangeAroundZero(hf[x], 0.132f);

        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void separateHf_Uhf(float* hf, float* uhf, int64_t width, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        separateHf_Uhf_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(hf, uhf, width);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void XybLowFreqToVals_Kernel(float* xplane, float* yplane, float* bplane, int64_t width){
        int64_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        //float3 old = makeFloat3(xplane[x], yplane[x], bplane[x]);
        XybLowFreqToVals(xplane[x], yplane[x], bplane[x]);
        //if ((x == 376098 && width == 1080*1920) || ((x == 59456 && width == 540*960))) printf("lfreqval : %f %f %f from %f %f %f\n", xplane[x], yplane[x], bplane[x], old.x, old.y, old.z); 
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void XybLowFreqToVals(float* xplane, float* yplane, float* bplane, int64_t width, hipStream_t stream){
        int64_t th_x = std::min((int64_t)256, width);
        int64_t bl_x = (width-1)/th_x + 1;
        XybLowFreqToVals_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(xplane, yplane, bplane, width);
        GPU_CHECK(hipGetLastError());
    }

    void separateFrequencies(float* src[3], float* temp[3], float* lf[3], float* mf[3], float* hf[2], float* uhf[2], int64_t width, int64_t height, GaussianHandle& gaussianHandle, hipStream_t stream){
        for (int i = 0; i < 3; i++){
            //we separate lf to get mf BUT we put mf on hf if i != 2 for later reasons
            blur(lf[i], src[i], temp[i], width, height, gaussianHandle, 4, stream);

            if (i == 2){
                //mf = blur(xyb-lf)
                subarray(src[i], lf[i], mf[i], width*height, stream);
                blur(mf[i], temp[i], width, height, gaussianHandle, 3, stream);
                break;
            }
            //mf (hf (uhf)) = xyb-lf //mf is stored on hf which is stored in uhf
            //same thing here, we will put the real hf into uhf to avoid later copy
            subarray(src[i], lf[i], uhf[i], width*height, stream);
            //mf = blur(mf (hf (uhf))) //we blur mf BUT mf is on hf which is on uhf. After this, mf is stored in mf but hf is on uhf
            //the real mf is blurred and we avoid the need to copy the unblurred mf to hf (uhf)
            blur(mf[i], uhf[i], temp[i], width, height, gaussianHandle, 3, stream);

            //hf (uhf) = op(mf, hf (uhf))
            if (i == 0){
                subarray_removerangearound0(mf[i], uhf[i], width*height, 0.29f, stream); 
            } else {
                subarray_amplifyrangearound0(mf[i], uhf[i], width*height, 0.1f, stream);
            }

        }
        //using uhf which contains hf in reality
        supressXbyY(uhf[0], uhf[1], width*height, 46.0f, stream);

        for (int i = 0; i < 2; i++){
            //original does uhf = hf but hf is already in uhf.
            //next is hf = blur(hf (uhf)) -> hf is now at its place and uhf has the old hf copy
            blurDstNoTemp(hf[i], uhf[i], width, height, gaussianHandle, 1, stream);

            if (i == 0){
                subarray_removerangearound0(hf[i], uhf[i], width*height, 1.5f, stream);
                removerangearound0(uhf[i], width*height, 0.04f, stream);
            } else {
                separateHf_Uhf(hf[i], uhf[i], width*height, stream);
            }
        }

        XybLowFreqToVals(lf[0], lf[1], lf[2], width*height, stream);        
    }
}