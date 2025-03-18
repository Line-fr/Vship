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
        const float scaler = s + (yw * (1.0 - s)) / (yw + yval * yval);
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

    __global__ void subarray_removerangearound0_Kernel(float* first, float* second, float width, float w){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        second[x] -= first[x];
        RemoveRangeAroundZero(first[x], w);
    }

    void subarray_removerangearound0(float* first, float* second, int width, float w, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        subarray_removerangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void removerangearound0_Kernel(float* first, float width, float w){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        RemoveRangeAroundZero(first[x], w);
    }

    void removerangearound0(float* first, int width, float w, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        removerangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void subarray_amplifyrangearound0_Kernel(float* first, float* second, float width, float w){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        second[x] -= first[x];
        AmplifyRangeAroundZero(first[x], w);
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void subarray_amplifyrangearound0(float* first, float* second, int width, float w, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        subarray_amplifyrangearound0_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, w);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void supressXbyY_Kernel(float* first, float* second, float width, float yw){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        supressXbyY(first[x], second[x], yw);
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void supressXbyY(float* first, float* second, int width, float yw, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        supressXbyY_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(first, second, width, yw);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void separateHf_Uhf_Kernel(float* lf, float* hf, float* uhf, float width){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        const float kMaxclampHf = 28.4691806922f;
        const float kMaxclampUhf = 5.19175294647f;

        MaximumClamp(hf[x], kMaxclampHf);
        uhf[x] -= hf[x];
        MaximumClamp(uhf[x], kMaxclampUhf);
        uhf[x] *= 2.69313763794;
        hf[x] *= 2.155;
        AmplifyRangeAroundZero(hf[x], 0.132f);

        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void separateHf_Uhf(float* lf, float* hf, float* uhf, int width, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        separateHf_Uhf_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(lf, hf, uhf, width);
        GPU_CHECK(hipGetLastError());
    }

    __global__ void XybLowFreqToVals_Kernel(float* xplane, float* yplane, float* bplane, int width){
        size_t x = threadIdx.x + blockIdx.x*blockDim.x;

        if (x >= width) return;

        XybLowFreqToVals(xplane[x], yplane[x], bplane[x]);
        //printf("res : %f and %f\n", first[x], second[x]);
    }

    void XybLowFreqToVals(float* xplane, float* yplane, float* bplane, int width, hipStream_t stream){
        int th_x = std::min(256, width);
        int bl_x = (width-1)/th_x + 1;
        XybLowFreqToVals_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(xplane, yplane, bplane, width);
        GPU_CHECK(hipGetLastError());
    }

    void separateFrequencies(Plane_d src[3], Plane_d temp[3], Plane_d lf[3], Plane_d mf[3], Plane_d hf[2], Plane_d uhf[2], float* gaussiankernel){
        int width = src[0].width; int height = src[0].height;
        hipStream_t stream = src[0].stream;
        
        for (int i = 0; i < 3; i++){
            //we separate lf to get mf BUT we put mf on hf if i != 2 for later reasons
            src[i].blur(lf[i], temp[i], 7.15593339443f, 0.0f, gaussiankernel);

            if (i == 2){
                //mf = blur(xyb-lf)
                subarray(src[i].mem_d, lf[i].mem_d, mf[i].mem_d, width*height, stream);
                mf[i].blur(temp[i], 3.22489901262f, 0.0f, gaussiankernel);
                break;
            }
            //mf (hf (uhf)) = xyb-lf //mf is stored on hf which is stored in uhf
            //same thing here, we will put the real hf into uhf to avoid later copy
            subarray(src[i].mem_d, lf[i].mem_d, uhf[i].mem_d, width*height, stream);
            //mf = blur(mf (hf (uhf))) //we blur mf BUT mf is on hf which is on uhf. After this, mf is stored in mf but hf is on uhf
            //the real mf is blurred and we avoid the need to copy the unblurred mf to hf (uhf)
            uhf[i].blur(mf[i], temp[i], 3.22489901262f, 0.0f, gaussiankernel);

            //hf (uhf) = op(mf, hf (uhf))
            if (i == 0){
                subarray_removerangearound0(mf[i].mem_d, uhf[i].mem_d, width*height, 0.29f, stream); 
            } else {
                subarray_amplifyrangearound0(mf[i].mem_d, uhf[i].mem_d, width*height, 0.1f, stream);
            }

        }
        //using uhf which contains hf in reality
        supressXbyY(uhf[0].mem_d, uhf[1].mem_d, width*height, 46.0f, stream);

        for (int i = 0; i < 2; i++){
            //original does uhf = hf but hf is already in uhf.
            //next is hf = blur(hf (uhf)) -> hf is now at its place and uhf has the old hf copy
            uhf[i].blur(hf[i], temp[i], 1.56416327805f, 0.0f, gaussiankernel);

            if (i == 0){
                subarray_removerangearound0(hf[i].mem_d, uhf[i].mem_d, width*height, 1.5f, stream);
                removerangearound0(uhf[i].mem_d, width*height, 0.04f, stream);
            } else {
                separateHf_Uhf(lf[i].mem_d, hf[i].mem_d, uhf[i].mem_d, width*height, stream);
            }
        }

        XybLowFreqToVals(lf[0].mem_d, lf[1].mem_d, lf[2].mem_d, width*height, stream);        
    }
}