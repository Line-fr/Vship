#pragma once

namespace cvvdp{

__device__ constexpr float kernel_a = 0.4f;
__device__ constexpr float gaussPyrKernel[5] = {0.25-kernel_a/2.,0.25, kernel_a, 0.25, 0.25-kernel_a/2.};

//separable, but not worth it for a kernel of size 5
__global__ void gaussPyrReduce_Kernel(float* dst, float* src, int64_t source_width, int64_t source_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    if (thid >= nw*nh) return;

    //in new space
    const int x = thid%nw;
    const int y = thid/nw;

    float nval = 0.f;

    for (int dx = -2; dx <= 2; dx++){
        const float kernel_x = gaussPyrKernel[dx+2];
        int ref_ind_x = 2*x + dx;
        //symmetric padding
        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x;
        if (ref_ind_x >= source_width) ref_ind_x = 2*source_width - ref_ind_x;
        for (int dy = -2; dy <= 2; dy++){
            const float kernel_y = gaussPyrKernel[dy+2];
            int ref_ind_y = 2*y + dy;
            //symmetric padding
            if (ref_ind_y < 0) ref_ind_y = -ref_ind_y;
            if (ref_ind_y >= source_height) ref_ind_y = 2*source_height - ref_ind_y;

            nval += kernel_x*kernel_y*src[ref_ind_y*source_width+ref_ind_x];
        }
    }

    dst[y*nw+x] = nval;
}

void gaussPyrReduce(float* dst, float* src, int64_t source_width, int64_t source_height, hipStream_t stream){
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    
    int th_x = 256;
    int64_t bl_x = (nw*nh+th_x-1)/th_x;
    gaussPyrReduce_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, source_width, source_height);
}

//separable, but not worth it for a kernel of size 5
template<bool subdst>
__global__ void gaussPyrExpand_Kernel(float* dst, float* src, int64_t new_width, int64_t new_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int ow = (new_width+1)/2;
    if (thid >= new_width*new_height) return;

    //in new space
    const int x = thid%new_width;
    const int y = thid/new_width;

    float nval = 0.f;

    int parity_x = x%2;
    int parity_y = y%2;

    for (int dx = -2+parity_x; dx <= 2; dx+=2){
        const float kernel_x = 2*gaussPyrKernel[dx+2];
        const int ref_ind_x = (x + dx)/2; //funny: x+dx is always even
        for (int dy = -2-parity_y; dy <= 2; dy+=2){
            const float kernel_y = 2*gaussPyrKernel[dy+2];
            const int ref_ind_y = (y+dy)/2; //(y+dy) is always even

            nval += kernel_x*kernel_y*src[ref_ind_y*ow+ref_ind_x];
        }
    }

    if constexpr (subdst) {
        dst[thid] -= nval;
    } else {
        dst[thid] = nval;
    }
}

template<bool subdst>
void gaussPyrExpand(float* dst, float* src, int64_t new_width, int64_t new_height, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    gaussPyrExpand_Kernel<subdst><<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, new_width, new_height);
}

template<bool isMean, int multiplier>
__global__ void baseBandPyrRefine_Kernel(float* p, float* Lbkg, int64_t width){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width) return;

    float val;
    if constexpr (!isMean){
        val = min(p[thid]/max(0.01f, Lbkg[thid]), 1000.f);
    } else {
        //if (thid == 0) printf("value %f %f\n", Lbkg[0], p[thid]);
        //then our adress is the mean, a single float at 0
        val = min(p[thid]/max(0.01f, Lbkg[0]), 1000.f);
    }
    p[thid] = val*multiplier;
}

//gets the contrast from the layers
template<bool isMean, int multiplier>
void baseBandPyrRefine(float* p, float* Lbkg, int64_t width, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (width + th_x-1)/th_x;
    baseBandPyrRefine_Kernel<isMean, multiplier><<<dim3(bl_x), dim3(th_x), 0, stream>>>(p, Lbkg, width);   
}

//pointer jumping, start with 1024 threads
__global__ void reduceSum(float* dst, float* src, int64_t size, bool divide){
    const int64_t global_thid = threadIdx.x + blockIdx.x * blockDim.x;
    const int local_thid = threadIdx.x;
    constexpr int threadnum = 1024;

    __shared__ float pointerJumpingBuffer[threadnum]; //one float per thread

    if (global_thid >= size){
        pointerJumpingBuffer[local_thid] = 0;
    } else {
        pointerJumpingBuffer[local_thid] = src[global_thid];
    }

    __syncthreads();

    int next = 1;
    while (next < threadnum){
        if (local_thid + next < threadnum && (local_thid%(next*2) == 0)){
            pointerJumpingBuffer[local_thid] += pointerJumpingBuffer[local_thid+next];
        }
        next *= 2;
        __syncthreads();
    }

    if (local_thid == 0){
        if (divide) pointerJumpingBuffer[0] /= size;
        dst[blockIdx.x] = pointerJumpingBuffer[0];
    }
}

//the result will be at temp[0]. We suppose that temp is of the same size as src
void computeMean(float* src, float* temp, int64_t size, hipStream_t stream){
    constexpr int th_x = 1024;
    int bl_x;

    float* final_dst = temp; //to contain temp[0]
    float* tempbuffer[3] = {temp+1, temp+1+(size+1023)/1024, src};
    int oscillator = 2; //corresponds to current source except the first time
    while (size > 1024){
        bl_x = (size+th_x-1)/th_x;
        int destination = (oscillator == 2) ? 0 : (oscillator^1);
        reduceSum<<<dim3(bl_x), dim3(th_x), 0, stream>>>(tempbuffer[destination], tempbuffer[oscillator], size, (oscillator == 2));
        oscillator = destination;
        size = (size+1023)/1024;
    }
    bl_x = 1;
    reduceSum<<<dim3(bl_x), dim3(th_x), 0, stream>>>(final_dst, tempbuffer[oscillator], size, (oscillator == 2));
}

std::vector<float> get_frequencies(const int64_t width, const int64_t height, const float ppd){
    const float min_freq = 0.2;
    const int maxLevel_forRes = std::log2(std::min(width, height))-1;
    const int maxLevel_forPPD = std::ceil(-std::log2(2*min_freq/0.3228/ppd));
    const int maxLevel_hard = 14;
    const int levels = std::min(maxLevel_forPPD, std::max(maxLevel_forRes, maxLevel_hard));

    std::vector<float> band_frequencies(levels);
    for (int i = 0; i < levels; i++){
        band_frequencies[i] = 0.3228*0.5*ppd/((float)(1 << i));
    }
    return std::move(band_frequencies);
}

class LpyrManager{
    std::vector<std::pair<int64_t, int64_t>> resolutions; //for each band
    std::vector<float*> adresses;
    int64_t planeOffset;
    std::vector<float> band_frequencies;
    float ppd;
public: 
    //plane contains 5 planes each twice the size
    //the last plane will contain L_bkg while the first planes should contain the 4 temporal channels for the first half
    LpyrManager(float* plane, const int64_t width, const int64_t height, const float ppd, int64_t bandOffset, const hipStream_t stream){
        band_frequencies = get_frequencies(width, height, ppd);
        const int levels = band_frequencies.size();

        planeOffset = bandOffset;

        resolutions.resize(levels);
        adresses.resize(levels);
        int64_t w = width;
        int64_t h = height;
        float* p = plane;
        for (int i = 0; i < levels; i++){
            resolutions[i].first = w;
            resolutions[i].second = h;
            adresses[i] = p;

            if (i != levels-1){
                //first is Y_sustained, it governs L_BKG
                gaussPyrReduce(p+w*h, p, w, h, stream);
                gaussPyrExpand<false>(p+4*planeOffset, p+w*h, w, h, stream);
                subarray(p, p+4*planeOffset, p, w*h, stream);
                if (i == 0){
                    baseBandPyrRefine<false, 1>(p, p+4*planeOffset, w*h, stream);
                } else {
                    baseBandPyrRefine<false, 2>(p, p+4*planeOffset, w*h, stream);
                }
                //then other channels
                for (int channel = 1; channel < 4; channel++){
                    //we first create the next step of the pyramid
                    gaussPyrReduce(p+w*h+channel*planeOffset, p+channel*planeOffset, w, h, stream);
                    //then we substract its upscaled version from the original to create the "layer"
                    gaussPyrExpand<true>(p+channel*planeOffset, p+w*h+channel*planeOffset, w, h, stream);
                    //then we transform this layer into a contrast by using the L_BKG computed before the loop
                    if (i == 0){
                        baseBandPyrRefine<false, 1>(p+channel*planeOffset, p+4*planeOffset, w*h, stream);
                    } else {
                        baseBandPyrRefine<false, 2>(p+channel*planeOffset, p+4*planeOffset, w*h, stream);
                    }
                }
            } else {
                //here Lbkg is different, it is the mean.
                //now we need to take the mean. 
                computeMean(p, p+4*planeOffset, w*h, stream);
                float* meanp = p+4*planeOffset;
                for (int channel = 0; channel < 4; channel++){
                    baseBandPyrRefine<true, 1>(p+channel*planeOffset, meanp, w*h, stream);
                }
            }

            p += w*h;
            w = (w+1)/2;
            h = (h+1)/2;
        }
    }
    int getSize(){
        return resolutions.size();
    }
    std::pair<int64_t, int64_t> getResolution(int i){
        return resolutions[i];
    }
    float getFrequency(int i){
        return band_frequencies[i];
    }
    float* getLbkg(int band){
        return adresses[band]+4*planeOffset;
    }
    float* getContrast(int channel, int band){
        return adresses[band]+channel*planeOffset;
    }
};

}