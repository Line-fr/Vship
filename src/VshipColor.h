#ifndef VSHIPCOLOR_API_HEADER
#define VSHIPCOLOR_API_HEADER

enum Vship_Sample_t{
    Vship_SampleFLOAT,
    Vship_SampleHALF,
    Vship_SampleUINT8,
    Vship_SampleUINT9,
    Vship_SampleUINT10,
    Vship_SampleUINT12,
    Vship_SampleUINT14,
    Vship_SampleUINT16,
};

enum Vship_Range_t{
    Vship_RangeLimited,
    Vship_RangeFull,
};

typedef struct {
    int subw = 0;
    int subh = 0;
} Vship_ChromaSubsample_t;

enum Vship_ChromaLocation_t{
    Vship_ChromaLoc_Top,
    Vship_ChromaLoc_TopLeft,
    Vship_ChromaLoc_Left,
    Vship_ChromaLoc_Center,
};

enum Vship_YUVMatrix_t{
    Vship_MATRIX_RGB = 0,
    Vship_MATRIX_BT709 = 1,
    Vship_MATRIX_BT470_BG = 5,
    Vship_MATRIX_BT2020_NCL = 9,
    Vship_MATRIX_BT2020_CL = 10,
    Vship_MATRIX_BT2100_ICTCP = 14,
};

enum Vship_TransferFunction_t{
    Vship_TRC_BT709 = 1,
    Vship_TRC_Linear = 8,
    Vship_TRC_sRGB = 13,
    Vship_TRC_PQ = 16,
    Vship_TRC_ST428 = 17,
    Vship_TRC_HLG = 18,
};

enum Vship_Primaries_t{
    Vship_PRIMARIES_BT709 = 1,
    Vship_PRIMARIES_BT2020 = 9,
};

typedef struct {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
} Vship_CropRectangle_t;

typedef struct {
    Vship_Sample_t sample = Vship_SampleUINT8;
    Vship_Range_t range = Vship_RangeLimited;
    Vship_ChromaSubsample_t subsampling = {1, 1};
    Vship_ChromaLocation_t chromaLocation = Vship_ChromaLoc_TopLeft;
    Vship_YUVMatrix_t YUVMatrix = Vship_MATRIX_BT709;
    Vship_TransferFunction_t transferFunction = Vship_TRC_BT709;
    Vship_Primaries_t primaries = Vship_PRIMARIES_BT709;
    Vship_CropRectangle_t crop = {0, 0, 0, 0};
} Vship_Colorspace_t;

#endif