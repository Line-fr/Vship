#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"

#include "parameters.hpp"
#include "display_models.hpp"

namespace cvvdp{

template <InputMemType T>
double CVVDPprocess(const uint8_t *dstp, int64_t dststride, const uint8_t *srcp1[3], const uint8_t *srcp2[3], int64_t stride, int64_t stride2, int64_t width, int64_t height, float fps, DisplayModel* model, int64_t maxshared, hipStream_t stream){
    std::cout << "fps: " << fps << std::endl;
    return 10.;
}

class CVVDPComputingImplementation{
    DisplayModel* model;
    float fps;
    int64_t width;
    int64_t height;
    int maxshared;
    hipStream_t stream;
public:
    void init(int64_t width, int64_t height, float fps, std::string model_key){
        this->width = width;
        this->height = height;
        this->fps = fps;

        model = new DisplayModel(model_key);

        hipStreamCreate(&stream);

        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);

        maxshared = devattr.sharedMemPerBlock;
    }
    void destroy(){
        delete model;
        hipStreamDestroy(stream);
    }
    template <InputMemType T>
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
        return CVVDPprocess<T>(dstp, dststride, srcp1, srcp2, stride, stride2, width, height, fps, model, maxshared, stream);
    }
};

}