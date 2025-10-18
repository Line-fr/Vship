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

    std::vector<float> res(size);
    for (int i = 0; i < size; i++){
        res[i] = 0;
        for (int k = 0; k < inp.size(); k++){
            res[i] += 2*inp[k]*std::cos(2*PI*k*i/size)/size; //2/N is normalization factor and exceptions are handled before
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
    float* convolutionKernel_d; //already prepared for GPU
public:
    int size;
    int fourrier_size;
    TemporalFilter(){
    }
    void init(float fps){
        size = (ceil(0.25*fps/2)*2)+1;
        fourrier_size = size/2 + 1; //only uses real FT
        
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
    float* getFilter(int j){ //returns gpu adresses
        //j must be 0, 1, 2 or 3. If not the pointer will be invalid
        //the size of the returned array is size
        return convolutionKernel_d + j*size;
    }
    void destroy(){
        hipFree(convolutionKernel_d);
    }
};



}