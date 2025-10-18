#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"

#include "parameters.hpp"
#include "display_models.hpp"
#include "temporalFilter.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, TemporalRing temporalRing, int64_t width, int64_t height, int64_t maxshared, hipStream_t stream){
    
    return 10.;
}

class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    TemporalRing tempFilter;
    int64_t width = 0;
    int64_t height = 0;
    int maxshared = 0;
    hipStream_t stream = 0;
public:
    void init(int64_t width, int64_t height, float fps, std::string model_key){
        this->width = width;
        this->height = height;
        this->fps = fps;

        tempFilter.init(fps, width, height);
        model = new DisplayModel(model_key);

        hipStreamCreate(&stream);

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;
    }
    void destroy(){
        tempFilter.destroy();
        delete model;
        hipStreamDestroy(stream);
    }
    template <InputMemType T>
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int64_t stride, int64_t stride2){

    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        loadImageToRing<T>(srcp1, srcp2, stride, stride2);
        return CVVDPprocess(dstp, dststride, tempFilter, width, height, maxshared, stream);
    }
};

}