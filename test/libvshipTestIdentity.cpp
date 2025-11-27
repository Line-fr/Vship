#include "VshipAPI.h"
#include<iostream>
#include<cstdint>
#include<chrono>
#include<thread>
#include<vector>
#include <array>
#include<cstring>
#include<ranges>

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetErrorMessage(err, errmsg, 1024);\
    printf("Vship Error occured: %s", errmsg);\
    exit(1);\
}

int bytesizeSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_SampleUINT8:
            return 1;
        case Vship_SampleUINT9:
        case Vship_SampleUINT10:
        case Vship_SampleUINT12:
        case Vship_SampleUINT14:
        case Vship_SampleUINT16:
        case Vship_SampleHALF:
            return 2;
        case Vship_SampleFLOAT:
            return 4;
    }
    return 0;
}

void printColorspace(Vship_Colorspace_t colorspace){
    std::cout << "Source Size: " << colorspace.width << "x" << colorspace.height << std::endl;
    std::cout << "Resize To: " << colorspace.target_width << "x" << colorspace.target_height << std::endl;
    std::cout << "Cropped by (Top/Bottom/Left/Right): " << colorspace.crop.top << "/" << colorspace.crop.bottom << "/" << colorspace.crop.left << "/" << colorspace.crop.right << std::endl;
    std::cout << "=> Converted Size: ";
    if (colorspace.target_width != -1){ 
        std::cout << colorspace.target_width-colorspace.crop.right-colorspace.crop.left;
    } else {
        std::cout << colorspace.width-colorspace.crop.right-colorspace.crop.left;
    }
    std::cout << "x";
    if (colorspace.target_height != -1){
        std::cout << colorspace.target_height-colorspace.crop.top-colorspace.crop.bottom;
    } else {
        std::cout << colorspace.height-colorspace.crop.top-colorspace.crop.bottom;
    }
    std::cout << std::endl;
    std::cout << "Sample Type: ";
    switch(colorspace.sample){
        case Vship_SampleUINT8:
            std::cout << "Uint8_t";
            break;
        case Vship_SampleUINT9:
            std::cout << "Uint9_t";
            break;
        case Vship_SampleUINT10:
            std::cout << "Uint10_t";
            break;
        case Vship_SampleUINT12:
            std::cout << "Uint12_t";
            break;
        case Vship_SampleUINT14:
            std::cout << "Uint14_t";
            break;
        case Vship_SampleUINT16:
            std::cout << "Uint16_t";
            break;
        case Vship_SampleHALF:
            std::cout << "Half";
            break;
        case Vship_SampleFLOAT:
            std::cout << "Float";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Color Family: ";
    if (colorspace.colorFamily == Vship_ColorRGB){
        std::cout << "RGB";
    } else if (colorspace.colorFamily == Vship_ColorYUV){
        std::cout << "YUV";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Range: ";
    if (colorspace.range == Vship_RangeLimited){
        std::cout << "Limited";
    } else if (colorspace.range == Vship_RangeFull){
        std::cout << "Full";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Subsampling (log): " << colorspace.subsampling.subw << "x" << colorspace.subsampling.subh << std::endl;
    std::cout << "Chroma Location: ";
    switch(colorspace.chromaLocation){
        case Vship_ChromaLoc_Center:
            std::cout << "Center";
            break;
        case Vship_ChromaLoc_TopLeft:
            std::cout << "Top-Left";
            break;
        case Vship_ChromaLoc_Left:
            std::cout << "Left";
            break;
        case Vship_ChromaLoc_Top:
            std::cout << "Top";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "YUV Matrix: ";
    switch(colorspace.YUVMatrix){
        case Vship_MATRIX_RGB:
            std::cout << "RGB";
            break;
        case Vship_MATRIX_BT709:
            std::cout << "BT709";
            break;
        case Vship_MATRIX_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_MATRIX_ST170_M:
            std::cout << "ST170_M";
            break;
        case Vship_MATRIX_BT2020_NCL:
            std::cout << "BT2020_NCL";
            break;
        case Vship_MATRIX_BT2020_CL:
            std::cout << "BT2020_CL";
            break;
        case Vship_MATRIX_BT2100_ICTCP:
            std::cout << "BT2100_ICTCP";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Transfer Function: ";
    switch(colorspace.transferFunction){
        case Vship_TRC_BT709:
            std::cout << "BT709";
            break;
        case Vship_TRC_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_TRC_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_TRC_BT601:
            std::cout << "BT601";
            break;
        case Vship_TRC_Linear:
            std::cout << "Linear";
            break;
        case Vship_TRC_sRGB:
            std::cout << "sRGB";
            break;
        case Vship_TRC_PQ:
            std::cout << "PQ";
            break;
        case Vship_TRC_ST428:
            std::cout << "ST428";
            break;
        case Vship_TRC_HLG:
            std::cout << "HLG";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Primaries: ";
    switch(colorspace.primaries){
        case Vship_PRIMARIES_INTERNAL:
            std::cout << "XYZ";
            break;
        case Vship_PRIMARIES_BT709:
            std::cout << "BT709";
            break;
        case Vship_PRIMARIES_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_PRIMARIES_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_PRIMARIES_BT2020:
            std::cout << "BT2020";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
}

enum Vship_Metric{
    SSIMULACRA2,
    BUTTERAUGLI,
    CVVDP,
};

const std::array<int64_t, 2> testWidth = {720, 1920};
const std::array<int64_t, 2> testHeight = {480, 1080};
const std::array<Vship_Metric, 3> testMetric = {SSIMULACRA2, BUTTERAUGLI, CVVDP};
const std::array<Vship_Sample_t, 7> testSample= {Vship_SampleFLOAT, Vship_SampleUINT16, Vship_SampleUINT14, Vship_SampleUINT12, Vship_SampleUINT10, Vship_SampleUINT9, Vship_SampleUINT8};
const std::array<Vship_Range_t, 2> testRange= {Vship_RangeLimited, Vship_RangeFull};
const std::array<Vship_ChromaSubsample_t, 6> testChromaSub = {Vship_ChromaSubsample_t({0, 0}), Vship_ChromaSubsample_t({1, 0}), Vship_ChromaSubsample_t({2, 0}), Vship_ChromaSubsample_t({0, 1}), Vship_ChromaSubsample_t({1, 1}), Vship_ChromaSubsample_t({2, 1})};
const std::array<Vship_ChromaLocation_t, 4> testChromaLoc = {Vship_ChromaLoc_Left, Vship_ChromaLoc_Center, Vship_ChromaLoc_Top, Vship_ChromaLoc_TopLeft};
const std::array<Vship_ColorFamily_t, 1> testColorFam = {Vship_ColorYUV};
const std::array<Vship_YUVMatrix_t, 7> testMatrix = {
    Vship_MATRIX_RGB,
    Vship_MATRIX_BT709,
    Vship_MATRIX_BT470_BG,
    Vship_MATRIX_ST170_M,
    Vship_MATRIX_BT2020_NCL,
    Vship_MATRIX_BT2020_CL,
    Vship_MATRIX_BT2100_ICTCP,
};
const std::array<Vship_TransferFunction_t, 9> testTransfer = {
    Vship_TRC_BT709,
    Vship_TRC_BT470_M,
    Vship_TRC_BT470_BG,
    Vship_TRC_BT601,
    Vship_TRC_Linear,
    Vship_TRC_sRGB,
    Vship_TRC_PQ,
    Vship_TRC_ST428,
    Vship_TRC_HLG,
};
const std::array<Vship_Primaries_t, 5> testPrimaries = {
    Vship_PRIMARIES_INTERNAL,
    Vship_PRIMARIES_BT709,
    Vship_PRIMARIES_BT470_M,
    Vship_PRIMARIES_BT470_BG,
    Vship_PRIMARIES_BT2020,
};
const std::array<Vship_CropRectangle_t, 3> testCrop = {Vship_CropRectangle_t({0, 0, 0, 0}), Vship_CropRectangle_t({10, 10, 10, 10}), Vship_CropRectangle_t({1, 3, 5, 7})};
const std::array<int64_t, 2> testTargetWidth = {-1, 720};
const std::array<int64_t, 2> testTargetHeight = {-1, 480};

uint64_t seed = 42;
//is a not so random thing I made because we are NOT using it for cryptography
uint64_t getRandomNumber(){
    seed = (4llu*seed) << 36 + (3llu*seed) << 24 + (2llu*seed) << 12 + (seed);
    seed = ((seed >> 32) << 32) + (seed >> 32);
    seed *= seed;
    return seed*4320984098llu;
}

void getRandomNumberSample(uint8_t* out, Vship_Sample_t sampleType){
    double n1, n2;
    float res;
    switch (sampleType){
        case Vship_SampleFLOAT:
            n1 = getRandomNumber();
            n2 = getRandomNumber();
            res = std::min(n1, n2)/std::max(n1, n2);
            *((float*)out) = res;
            return;
        case Vship_SampleUINT8:
            *out = getRandomNumber()%256llu;
            return;
        case Vship_SampleUINT9:
        case Vship_SampleUINT10:
        case Vship_SampleUINT12:
        case Vship_SampleUINT14:
        case Vship_SampleUINT16:
            *((uint16_t*)out) = getRandomNumber()%(1llu << bytesizeSample(sampleType));
            return;
    }
}

class Image{
private:
    uint8_t* srcp[3] = {NULL, NULL, NULL};
public:
    const uint8_t* csrcp[3];
    int64_t lineSize[3];
    Vship_Colorspace_t colorspace;
    Image(Vship_Colorspace_t colorspace) : colorspace(colorspace){
        const int64_t plane_heights[3] = {colorspace.height, colorspace.height / (1 << colorspace.subsampling.subh), colorspace.height / (1 << colorspace.subsampling.subh)};
        const int64_t plane_widths[3] = {colorspace.width, colorspace.width / (1 << colorspace.subsampling.subw), colorspace.width / (1 << colorspace.subsampling.subw)};

        lineSize[0] = bytesizeSample(colorspace.sample)*colorspace.width;
        lineSize[1] = lineSize[0] / (1 << colorspace.subsampling.subw);
        lineSize[2] = lineSize[0] / (1 << colorspace.subsampling.subw);
        lineSize[0] += 47; //add a randomm stride
        lineSize[1] += 38; //add a randomm stride
        lineSize[2] += 97; //add a randomm stride

        for (int i = 0; i < 3; i++){
            srcp[i] = (uint8_t*)malloc(lineSize[i]*plane_heights[i]);
            csrcp[i] = srcp[i];
        }

        //fill the image
        for (int plane = 0; plane < 3; plane++){
            for (int64_t x = 0; x < plane_widths[plane]; x++){
                for (int64_t y = 0; y < plane_heights[plane]; y++){
                    getRandomNumberSample(srcp[plane]+(y*plane_widths[plane]+x)*bytesizeSample(colorspace.sample), colorspace.sample);
                }
            }
        }
    }
    Image(Image& im){
        colorspace = im.colorspace;
        const int64_t plane_heights[3] = {colorspace.height, colorspace.height / (1 << colorspace.subsampling.subh), colorspace.height / (1 << colorspace.subsampling.subh)};
        for (int plane = 0; plane < 3; plane++){
            lineSize[plane] = im.lineSize[plane];
            srcp[plane] = (uint8_t*)malloc(lineSize[plane]*plane_heights[plane]);
            csrcp[plane] = srcp[plane];
            memcpy(srcp[plane], im.csrcp[plane], lineSize[plane]*plane_heights[plane]);
        }
        randomizeStride();
    }
    ~Image(){
        for (int i = 0; i < 3; i++){
            if (srcp[i] != NULL){
                free(srcp[i]);
                srcp[i] = NULL;
            }
        }
    }
    void randomizeStride(){
        const int64_t plane_heights[3] = {colorspace.height, colorspace.height / (1 << colorspace.subsampling.subh), colorspace.height / (1 << colorspace.subsampling.subh)};
        const int64_t plane_widths[3] = {colorspace.width, colorspace.width / (1 << colorspace.subsampling.subw), colorspace.width / (1 << colorspace.subsampling.subw)};
        for (int plane = 0; plane < 3; plane++){
            for (int64_t x = plane_widths[plane]*bytesizeSample(colorspace.sample); x < lineSize[plane]; x++){
                for (int64_t y = 0; y < plane_heights[plane]; y++){
                    srcp[plane][y*lineSize[plane]+x] = getRandomNumber()%256llu;
                }
            }
        }
    }
};

class MetricComparator{
    Vship_SSIMU2Handler ssimu2handler;
    Vship_ButteraugliHandler butterhandler;
    Vship_CVVDPHandler cvvdphandler;
public:
    MetricComparator(){

    }
    double compare(Vship_Colorspace_t colorspace, Vship_Metric metric){
        Image im1(colorspace);
        Image im2(im1);

        double res;
        Vship_ButteraugliScore butterres;
        switch (metric){
            case SSIMULACRA2:
                ErrorCheck(Vship_SSIMU2Init(&ssimu2handler, colorspace, colorspace));
                ErrorCheck(Vship_ComputeSSIMU2(ssimu2handler, &res, im1.csrcp, im2.csrcp, im1.lineSize, im2.lineSize));
                ErrorCheck(Vship_SSIMU2Free(ssimu2handler));
                res = (100. - res)/10.; //we expect a butteraugli like scale
                break;
            case BUTTERAUGLI:
                ErrorCheck(Vship_ButteraugliInit(&butterhandler, colorspace, colorspace, 5, 165.f));
                ErrorCheck(Vship_ComputeButteraugli(butterhandler, &butterres, NULL, 0, im1.csrcp, im2.csrcp, im1.lineSize, im2.lineSize));
                ErrorCheck(Vship_ButteraugliFree(butterhandler));
                res = butterres.norminf;
                break;
            case CVVDP:
                ErrorCheck(Vship_CVVDPInit(&cvvdphandler, colorspace, colorspace, 48, false, "standard_hdr_pq"));
                ErrorCheck(Vship_ComputeCVVDP(cvvdphandler, &res, NULL, 0, im1.csrcp, im2.csrcp, im1.lineSize, im2.lineSize));
                ErrorCheck(Vship_CVVDPFree(cvvdphandler));
                res = (10. - res); //we expect a scale butteraugli like
                break;
        }
        return res;
    }
    ~MetricComparator(){
    }
};

int main(){

    Vship_Version vshipVer = Vship_GetVersion();
    std::cout << "Running Vship Version " << vshipVer.major << "." << vshipVer.minor << "." << vshipVer.minorMinor << std::endl;
    std::cout << "Under Backend ";
    if (vshipVer.backend == Vship_Cuda) {
        std::cout << "Cuda";
    } else {
        std::cout << "HIP";
    }
    std::cout << std::endl;

    ErrorCheck(Vship_GPUFullCheck(0));
    Vship_DeviceInfo devinfo;
    ErrorCheck(Vship_GetDeviceInfo(&devinfo, 0));
    std::cout << "Running on GPU: " << devinfo.name << std::endl;

    MetricComparator comp;
    Vship_Colorspace_t c_test;
    auto init = std::chrono::high_resolution_clock::now();

    //elements we won't iterate over
    c_test.colorFamily = testColorFam[0];
    c_test.YUVMatrix = testMatrix[0];
    c_test.transferFunction = testTransfer[0];
    c_test.primaries = testPrimaries[0];

    uint64_t spaceSize = testWidth.size()*testTargetWidth.size()*testSample.size()*testRange.size()*testChromaSub.size()*testChromaLoc.size()*testCrop.size()*testMetric.size();
    uint64_t index = 0;

    std::cout << "0%" << std::flush;

    //input colorspace stress test
    for (int widthind = 0; widthind < testWidth.size(); widthind++){
        c_test.width = testWidth[widthind];
        c_test.height = testHeight[widthind];
        for (int widthtargetind = 0; widthtargetind < testTargetWidth.size(); widthtargetind++){
            c_test.target_width = testTargetWidth[widthtargetind];
            c_test.target_height = testTargetHeight[widthtargetind];
            
            auto product_iterator = std::ranges::cartesian_product_view(testSample, testRange, testChromaSub, testChromaLoc, testCrop, testMetric);
            for (const auto& [sample, range, chromasub, chromaloc, crop, metric]: product_iterator){
                c_test.sample = sample;
                c_test.range = range;
                c_test.subsampling = chromasub;
                c_test.chromaLocation = chromaloc;
                c_test.crop = crop;

                double res = comp.compare(c_test, metric);

                index++;
                if ((index-1)*1000/spaceSize != index*1000/spaceSize){
                    std::cout << "\r" << (float)index*100./spaceSize << "%" << std::flush;
                }

                if (res != 0.){
                    printColorspace(c_test);
                    std::cout << "Error. Non 0 result for this colorspace" << std::endl;
                    std::cout << "using metric: ";
                    if (metric == CVVDP) std::cout << "CVVDP";
                    if (metric == BUTTERAUGLI) std::cout << "Butteraugli";
                    if (metric == SSIMULACRA2) std::cout << "SSIMULACRA2";
                    std::cout << std::endl;
                    return 0;
                }
            }
        }
    }

    auto fin = std::chrono::high_resolution_clock::now();
    uint64_t millitaken = std::chrono::duration_cast<std::chrono::milliseconds>(fin - init).count();
    std::cout << "No error, Completed Test in " << millitaken/1000. << "s" << std::endl;
    return 0;
}