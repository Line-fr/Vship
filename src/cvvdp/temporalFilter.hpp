#pragma once

namespace cvvdp{

//will write on its input but put it back as before
//if inp is of size less than N/2, remaining values are counted as 0
std::vector<float> inverseRealFourrierTransform(std::vector<float> inp, int size){
    //ahead normalization process for reversing a FFT
    if ((int)inp.size() > size/2){
        inp[size/2] /= 2;
    }
    inp[0] /= 2;

    std::vector<float> res(size, 0);
    for (int i = 0; i < size; i++){
        for (int k = 0; k < inp.size(); k++){
            //we offset the index of res to get frequencies starting from 0 then positive then negative
            res[(i+inp.size()-1)%size] += 2*inp[k]*std::cos(2*PI*k*i/size)/size; //2/N is normalization factor and exceptions are handled before
        }
    }

    //putting back inp as before, except if inp values were reaching float lower limits, it should be lossless
    if ((int)inp.size() > size/2){
        inp[size/2] *= 2;
    }
    inp[0] *= 2;

    return res;
}

class TemporalFilter{
    float* convolutionKernel_d = NULL; //already prepared for GPU
public:
    int size = 0;
    TemporalFilter(){
    }
    void init(float fps){
        size = (ceil(0.25*fps/2)*2)+1;
        size = max(size, 1);
        int fourrier_size = size/2 + 1; //only uses real FT
        
        std::vector<float> freq(fourrier_size);
        for (int i = 0; i < fourrier_size; i++) freq[i] = fps*i/2;

        //Y sustained, rg sustained, yv sustained, Y transient
        std::vector<float> filters_fftdomain[4];
        //sustained channels
        for (int j = 0; j < 3; j++){
            filters_fftdomain[j].resize(fourrier_size);
            for (int i = 0; i < fourrier_size; i++){
                filters_fftdomain[j][i] = std::exp(-std::pow(freq[i], beta_tf[j])/sigma_tf[j]);
            }
        }
        //transient
        filters_fftdomain[3].resize(fourrier_size);
        for (int i = 0; i < fourrier_size; i++){
            //5 is the peak frequency of the transient which is a pass band
            float temp = std::pow(freq[i], beta_tf[3]) - std::pow(5, beta_tf[3]);
            temp *= temp; //square
            filters_fftdomain[3][i] = std::exp(-temp/sigma_tf[3]);
        }

        //go back from the FFT world, this will become a convolution kernel instead of a frequency multiplicator
        std::vector<float> temporal_filters[4];
        for (int j = 0; j < 4; j++) temporal_filters[j] = inverseRealFourrierTransform(filters_fftdomain[j], size);

        hipError_t erralloc = hipMalloc(&convolutionKernel_d, sizeof(float)*size*4);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }

        for (int j = 0; j < 4; j++){
            hipMemcpyHtoD(convolutionKernel_d + j*size, temporal_filters[j].data(), sizeof(float)*size);
        }
    }
    __device__ __host__ float* getFilter(int j){ //returns gpu adresses
        //j must be 0, 1, 2 or 3. If not the pointer will be invalid
        //the size of the returned array is size
        return convolutionKernel_d + j*size;
    }
    void destroy(){
        hipFree(convolutionKernel_d);
    }
};

//will contain the temporal image frame and manage them. 
// It will be able to apply the temporal filter to write 4 planes that corresponds to the 4 temporally filtered channels
class TemporalRing{
    float* internal_memory_d = NULL;
    //to get index of frame i, you need to do internal_memory+(3*plane_size)*((ind0+i)%max_temporal_size)
    //3* is because the 3 color planes of a single plane are stored together.
    int ind0 = 0; //index of current frame, past frames are stored after it up to temporal_size frames.
public:
    int temporal_size = 0;
    TemporalFilter tempFilterPreprocessor;
    int max_temporal_size = 0;
    int64_t width = -1;
    int64_t height = -1;
    TemporalRing(){}
    void init(float fps, int64_t width, int64_t height){
        this->width = width;
        this->height = height;
        tempFilterPreprocessor.init(fps);
        const int num_channel = 3;
        max_temporal_size = tempFilterPreprocessor.size;
        //std::cout << "allocationBytes temporalRing : " << sizeof(float)*width*height*max_temporal_size*num_channel << " for temporal buffer of size " << max_temporal_size << std::endl;
        hipError_t erralloc = hipMalloc(&internal_memory_d, sizeof(float)*width*height*max_temporal_size*num_channel);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }
    }
    //the pointer has all 3 planes next to each others
    //i is "how old" is the frame. 0 is current, 1 is previous,...
    __device__ __host__ float* getFramePointer(int i){
        assert(temporal_size > 0);
        //if i asked is too old, CVVDP goes to the oldest frame in store
        if (i >= temporal_size) i = temporal_size-1;
        return internal_memory_d+(3*width*height)*((ind0+i)%max_temporal_size);
    }
    //the ring suppose the current frame has been written
    //after rotation, frame 0 needs to be written too to be valid
    //this is done using getFramePointer(0) and writing the frame on the pointer
    void rotate(){
        //we increment temporal_size to signify we have one more frame in store now
        //if we reached max, it means the oldest one will be overwritten (intended)
        temporal_size = std::min(max_temporal_size, temporal_size+1);
        ind0 = (ind0 + max_temporal_size - 1)%max_temporal_size; //force maintain positive modulo
    }
    void reset(){
        temporal_size = 0;
    }
    void destroy(){
        hipFree(internal_memory_d);
        tempFilterPreprocessor.destroy();
    }
};

__global__ void temporalConvolutionKernel_d(TemporalRing ring, float* Y_sustained, float* RG_sustained, float* YV_sustained, float* Y_transient){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t planeSize = ring.width*ring.height;
    if (x >= planeSize) return;

    if (ring.temporal_size == 1){
        const float* srcptr = ring.getFramePointer(0);
        Y_sustained[x] = srcptr[x];
        RG_sustained[x] = srcptr[x+planeSize];
        YV_sustained[x] = srcptr[x+2*planeSize];
        Y_transient[x] = 0.f;
        return;
    }

    //we use float2 to use double issue float ALU computing units.
    float2 valueTemp;

    float2 kernelTemp;

    float2 resY_Y = {0.f, 0.f};
    float2 resRG_YV = {0.f, 0.f};
    //convolution over temporal dimension
    for (int k = 0;  k < ring.max_temporal_size; k++){
        //kth frame
        float* srcptr = ring.getFramePointer(k);

        kernelTemp.x = ring.tempFilterPreprocessor.getFilter(0)[k];
        kernelTemp.y = ring.tempFilterPreprocessor.getFilter(3)[k];
        valueTemp.x = srcptr[x];
        valueTemp.y = valueTemp.x;

        resY_Y = fmaf(valueTemp, kernelTemp, resY_Y);

        kernelTemp.x = ring.tempFilterPreprocessor.getFilter(1)[k];
        kernelTemp.y = ring.tempFilterPreprocessor.getFilter(2)[k];
        valueTemp.x = srcptr[x + planeSize];
        valueTemp.y = srcptr[x + planeSize*2];

        resRG_YV = fmaf(valueTemp, kernelTemp, resRG_YV);
    }
    Y_sustained[x] = resY_Y.x;
    Y_transient[x] = resY_Y.y;
    RG_sustained[x] = resRG_YV.x;
    YV_sustained[x] = resRG_YV.y;

    //if (x == 0) printf("after temporalFilter: %f %f %f %f\n", resY_Y.x, resRG_YV.x, resRG_YV.y, resY_Y.y);
}

//give it planes, it will overwrite with the 4 temporal channels
void computeTemporalChannels(TemporalRing& ring, float* Y_sustained, float* RG_sustained, float* YV_sustained, float* Y_transient, hipStream_t stream){
    int th_x = 256;
    int bl_x = (ring.width*ring.height+th_x-1)/th_x;
    temporalConvolutionKernel_d<<<dim3(bl_x), dim3(th_x), 0, stream>>>(ring, Y_sustained, RG_sustained, YV_sustained, Y_transient);
}


}