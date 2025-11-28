#pragma once

#include "csf_lut_weber.hpp"

namespace cvvdp{

//takes an ordered array of size 32 and a value and search the biggest index of element smaller
//(except if x is smaller, then it returns 0)
__device__ __host__ int indexSearch(const float xp[32], float x){
    for (int i = 1; i < 32; i++){
        if (xp[i] > x){
            return i-1;
        }
    }
    return 31;
}

class CSF_Handler{
    //on gpu
    float* mem_d; //one allocation, we free at destroy
    float* log_Lbkg_LUTIndex; //size 32 floats
    float* logS_r_LUT; //for each band, for each channel, 32 floats
public:
    void init(const int64_t width, const int64_t height, const float ppd){
        std::vector<float> band_frequencies = get_frequencies(width, height, ppd);
        const int levels = band_frequencies.size();
        hipError_t erralloc = hipMalloc(&mem_d, sizeof(float)*(1+4*levels)*32);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        log_Lbkg_LUTIndex = mem_d;

        float log_Lbkg_LUTIndex_CPU[32]; //load log
        float LUT_logrho[32];
        for (int i = 0; i < 32; i++){
            LUT_logrho[i] = std::log10(CSF_LUT::rho[i]);
            log_Lbkg_LUTIndex_CPU[i] = std::log10(CSF_LUT::L_bkg[i]);
        }
        GPU_CHECK(hipMemcpyHtoD(log_Lbkg_LUTIndex, log_Lbkg_LUTIndex_CPU, sizeof(float)*32));

        //we create the logS_r_LUT on CPU since they are relatively small and not worth sending;
        for (int band = 0; band < levels; band++){
            float logrho = std::log10(band_frequencies[band]);
            const int x0 = std::min(30, indexSearch(LUT_logrho, logrho));
            const float slope = (logrho - LUT_logrho[x0])/(LUT_logrho[x0+1] - LUT_logrho[x0]);
            for (int channel = 0; channel < 4; channel++){
                float res[32];
                //std::cout << "chan " << channel << " bandrho " << band_frequencies[band] << " result ";
                for (int i = 0; i < 32; i++){
                    const float y0i = CSF_LUT::D2LUT[channel][i][x0];
                    const float y1i = CSF_LUT::D2LUT[channel][i][x0+1];
                    res[i] = y0i + (y1i - y0i)*slope;
                    //std::cout << res[i] << " ";
                }
                //std::cout << std::endl;

                GPU_CHECK(hipMemcpyHtoD(mem_d+32*(1+band*4+channel), res, sizeof(float)*32));
            }
        }
    }
    void destroy(){
        GPU_CHECK(hipFree(mem_d));
    }
    //src is Lbkg of ref
    __device__ float computeSensitivityGPU(float src, int band, int channel){
        const float* LUTx = log_Lbkg_LUTIndex;
        const float* LUTy = mem_d+32*(1+band*4+channel);
        const float x = log10(src);
        float frac = 31*(x - LUTx[0])/(LUTx[31] - LUTx[0]);
        const int imin = frac;
        const int imax = min(31, imin+1);
        frac -= imin;
        /*
        const int imin = indexSearch(LUTx, x);
        const int imax = min(31, imin+1);
        const float yx0 = LUTx[imin];
        const float yx1 = LUTx[imax];
        float frac;
        if (imin == imax) {
            frac = 0.;
        } else {
            frac = (x - yx0)/(yx1-yx0);
        }
        */
        const float logS = LUTy[imin] * frac + LUTy[imax] * (1.f-frac) + sensitivity_correction/20.f;
        //if (threadIdx.x + blockIdx.x == 0) printf("SensitivityCompute: inp log: %f, imin %d imax %d frac %f -> logS %f, -> S %f\n", x, imin, imax, frac, logS, powf(10, logS));
        return powf(10, logS);
    }
};

}