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
__global__ void gaussPyrExpand_Kernel(float* dst, float* src, int64_t new_width, int64_t new_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int ow = (new_width+1)/2;
    if (thid >= new_width*new_height) return;

    //in new space
    const int x = thid%new_width;
    const int y = thid/new_height;

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

    dst[y*new_width+x] = nval;
}

void gaussPyrExpand(float* dst, float* src, int64_t new_width, int64_t new_height, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    gaussPyrExpand_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, new_width, new_height);
}

//separable, but not worth it for a kernel of size 5
__global__ void gaussPyrExpandSub_Kernel(float* dst, float* src, int64_t new_width, int64_t new_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int ow = (new_width+1)/2;
    if (thid >= new_width*new_height) return;

    //in new space
    const int x = thid%new_width;
    const int y = thid/new_height;

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

    dst[y*new_width+x] -= nval;
}

//instead of storing the result, it substracts it to dst
void gaussPyrExpandSub(float* dst, float* src, int64_t new_width, int64_t new_height, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    gaussPyrExpandSub_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, new_width, new_height);
}

__global__ void baseBandPyrRefine_Kernel(float* p, float* Lbkg, int64_t width){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width) return;

    p[thid] = min(p[thid]/max(0.01f, Lbkg[thid]), 1000.f);
}

//gets the contrast from the layers
void baseBandPyrRefine(float* p, float* Lbkg, int64_t width, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (width + th_x-1)/th_x;
    baseBandPyrRefine_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(p, Lbkg, width);   
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
    LpyrManager(float* plane, const int64_t width, const int64_t height, const float ppd, const hipStream_t stream){
        const float min_freq = 0.2;
        const int maxLevel_forRes = std::log2(std::min(width, height))-1;
        const int maxLevel_forPPD = std::ceil(-std::log2(2*min_freq/0.3228/ppd));
        const int maxLevel_hard = 14;
        const int levels = std::min(maxLevel_forPPD, std::max(maxLevel_forRes, maxLevel_hard));

        planeOffset = 2*width*height;

        resolutions.resize(levels);
        band_frequencies.resize(levels);
        adresses.resize(levels);
        int64_t w = width;
        int64_t h = height;
        float* p = plane;
        for (int i = 0; i < levels; i++){
            resolutions[i].first = w;
            resolutions[i].second = h;
            adresses[i] = p;

            band_frequencies[i] = 0.3228*0.5*ppd/((float)(1 << i));

            if (i != levels-1){
                //first is Y_sustained, it governs L_BKG
                gaussPyrReduce(p+w*h, p, w, h, stream);
                gaussPyrExpand(p+4*planeOffset, p+w*h, w, h, stream);
                subarray(p, p+4*planeOffset, p, w*h, stream);
                baseBandPyrRefine(p, p+4*planeOffset, w*h, stream);
                //then other channels
                for (int channel = 1; channel < 4; channel++){
                    //we first create the next step of the pyramid
                    gaussPyrReduce(p+w*h+channel*planeOffset, p+channel*planeOffset, w, h, stream);
                    //then we substract its upscaled version from the original to create the "layer"
                    gaussPyrExpandSub(p+channel*planeOffset, p+w*h+channel*planeOffset, w, h, stream);
                    //then we transform this layer into a contrast by using the L_BKG computed before the loop
                    baseBandPyrRefine(p+channel*planeOffset, p+4*planeOffset, w*h, stream);
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