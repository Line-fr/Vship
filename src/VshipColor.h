#ifndef VSHIPCOLOR_API_HEADER
#define VSHIPCOLOR_API_HEADER

enum Vship_Sample_t{
    Vship_SampleFLOAT = 0,
    Vship_SampleHALF = 1,
    Vship_SampleUINT8 = 2,
    Vship_SampleUINT9 = 3,
    Vship_SampleUINT10 = 5,
    Vship_SampleUINT12 = 7,
    Vship_SampleUINT14 = 9,
    Vship_SampleUINT16 = 11,
};

enum Vship_Range_t{
    Vship_RangeLimited = 0,
    Vship_RangeFull = 1,
};

typedef struct {
    int subw = 0;
    int subh = 0;
} Vship_ChromaSubsample_t;

enum Vship_ChromaLocation_t{
    Vship_ChromaLoc_Left = 0,
    Vship_ChromaLoc_Center = 1,
    Vship_ChromaLoc_TopLeft = 2,
    Vship_ChromaLoc_Top = 3,
};

enum Vship_YUVMatrix_t{
    Vship_MATRIX_RGB = 0,
    Vship_MATRIX_BT709 = 1,
    Vship_MATRIX_BT470_BG = 5,
    Vship_MATRIX_ST170_M = 6, //same as 5
    Vship_MATRIX_BT2020_NCL = 9,
    Vship_MATRIX_BT2020_CL = 10,
    Vship_MATRIX_BT2100_ICTCP = 14,
};

enum Vship_TransferFunction_t{
    Vship_TRC_BT709 = 1,
    Vship_TRC_BT470_M = 4,
    Vship_TRC_BT470_BG = 5,
    Vship_TRC_BT601 = 6, //same as 5
    Vship_TRC_Linear = 8,
    Vship_TRC_sRGB = 13,
    Vship_TRC_PQ = 16,
    Vship_TRC_ST428 = 17,
    Vship_TRC_HLG = 18,
};

enum Vship_Primaries_t{
    Vship_PRIMARIES_INTERNAL = -1, //corresponds to XYZ really
    Vship_PRIMARIES_BT709 = 1,
    Vship_PRIMARIES_BT470_M = 4,
    Vship_PRIMARIES_BT470_BG = 5,
    Vship_PRIMARIES_BT2020 = 9,
};

typedef struct {
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
} Vship_CropRectangle_t;

typedef struct {
    int64_t width;
    int64_t height;
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