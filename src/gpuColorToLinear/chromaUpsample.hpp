#pragma once

namespace VshipColorConvert{

//defined in resize.hpp
/*
class CubicHermitSplineInterpolator{
    float v1; float v2; float v3; float v4;
public:
    __device__ __host__ CubicHermitSplineInterpolator(const float p0, const float m0, const float p1, const float m1){
        v1 = 2*p0 + m0 - 2*p1 + m1;
        v2 = -3*p0 + 3*p1 - 2*m0 - m1;
        v3 = m0;
        v4 = p0;
    }
    __device__ __host__ float get(const float t){ //cubic uses t between 0 and 1
        float res = v1;
        res *= t;
        res += v2;
        res *= t;
        res += v3;
        res *= t;
        res += v4;
        return res;
    }
};*/

__device__ CubicHermitSplineInterpolator getHorizontalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height-1);
    x = min(x, width-1); //can be -1

    const float elm1 = (x <= 0) ? src[y*width] : src[y*width+x-1];
    const float el0 = (x < 0) ? elm1 : src[y*width+x];
    const float el1 = (x >= width-1) ? el0 : src[y*width+x+1];
    const float el2 = (x >= width-2) ? el1 : src[y*width+x+2];

    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

__device__ CubicHermitSplineInterpolator getVerticalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height-1); //can be equal to -1
    x = min(x, width-1);

    const float elm1 = (y <= 0) ? src[x] : src[(y-1)*width+x];
    const float el0 = (y < 0) ? elm1 : src[y*width+x];
    const float el1 = (y >= height-1) ? el0 : src[(y+1)*width+x];
    const float el2 = (y >= height-2) ? el1 : src[(y+2)*width+x];
    
    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

//block x should range from 0 to smallwidth INCLUDED
//dst of size width*height while src is of size smallwidth*height
__global__ void bicubicHorizontalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    x--;
    const int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const int64_t smallwidth = (width-1)/2+1;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height){
        if (x != -1 && 2*x+1 < width) dst[y*width + 2*x+1] = interpolator.get(0.25f);
        if (2*x+2 < width) dst[y*width + 2*x+2] = interpolator.get(0.75f);
    }
}

//block x should range from 0 to smallwidth-1 INCLUDED
//dst of size width*height while src is of size smallwidth*height
__global__ void bicubicHorizontalLeftUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const int64_t smallwidth = (width-1)/2+1;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0 and 0.5 (0 is directly our value)
    if (y < height){
        if (x*2 < width) dst[y*width + 2*x] = src[y*smallwidth + x];
        if (x*2+1 < width) dst[y*width + 2*x+1] = interpolator.get(0.5f);
        //if (x == 0 && y == 0) printf("in left chroma upscale we get for middle : %f\n", interpolator.get(0.5f));
    }
}

//block x should range from 0 to smallwidth INCLUDED
//dst of size width*height while src is of size smallwidth*height
__global__ void bicubicHorizontalCenterUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    x--;
    const int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const int64_t  smallwidth = (width-1)/4+1;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.125, 0.375, 0.625 and 0.875
    if (y < height){
        if (x != -1 && 4*x+2 < width) dst[y*width + 4*x+2] = interpolator.get(0.125f);
        if (x != -1 && 4*x+3 < width) dst[y*width + 4*x+3] = interpolator.get(0.375f);
        if (4*x+4 < width) dst[y*width + 4*x+4] = interpolator.get(0.625f);
        if (4*x+5 < width) dst[y*width + 4*x+5] = interpolator.get(0.875f);
    }
}

//block x should range from 0 to smallwidth-1 INCLUDED
//dst of size width*height while src is of size smallwidth*height
__global__ void bicubicHorizontalLeftUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const int64_t  smallwidth = (width-1)/4+1;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0, 0.25, 0.5 and 0.75 (0 is directly our value)
    if (y < height){
        if (4*x < width) dst[y*width + 4*x] = src[y*smallwidth + x];
        if (4*x+1 < width) dst[y*width + 4*x+1] = interpolator.get(0.25f);
        if (4*x+2 < width) dst[y*width + 4*x+2] = interpolator.get(0.5f);
        if (4*x+3 < width)  dst[y*width + 4*x+3] = interpolator.get(0.75f);
    }
}

//block y should range from 0 to smallheight INCLUDED
//dst of size width*height while src is of size width*smallheight
__global__ void bicubicVerticalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    y--;
    const int64_t smallheight = (height-1)/2+1;
    CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (x < width){
        //if (x == 1638 && y == -1) printf("at place y = %lld, x = %lld we put : %f from %f %f %f, interpol 0: %f\n", 2*y+2, x, interpolator.get(0.75f), src[(y+1)*width+x], src[(y+2)*width+x], src[(y+3)*width+x], interpolator.get(0.f));
        if (y != -1 && 2*y+1 < height) dst[(2*y +1)*width + x] = interpolator.get(0.25f);
        if (2*y+2 < height) dst[(2*y+2)*width + x] = interpolator.get(0.75f);
    }
}

//block y should range from 0 to smallheight-1 INCLUDED
//dst of size width*height while src is of size width*smallheight
__global__ void bicubicVerticalTopUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const int64_t smallheight = (height-1)/2+1;
    CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Top so we are interested in values: 0 and 0.5
    if (x < width){
        if (2*y < height) dst[(2*y)*width + x] = src[y*width + x];
        if (2*y+1 < height) dst[(2*y+1)*width + x] = interpolator.get(0.5f);
    }
}

//temp is of size 2 planes
//source is of size width * height possibly chroma downsampled
__host__ int inline upsample(float* temp, float* src[3], int64_t width, int64_t height, Vship_ChromaLocation_t location, Vship_ChromaSubsample_t subsampledata, hipStream_t stream){
    const int subw = subsampledata.subw;
    const int subh = subsampledata.subh;
    
    if (subw == 0 && subh == 0) return 0;
    int64_t smallwidth = ((width-1) >> subw)+1;
    int64_t smallheight = ((height-1) >> subh)+1;
    const int thx = 16;
    const int thy = 16;
    int blx = (smallwidth+1 + thx-1)/thx; //will change after upsampling of horizontal
    int bly = (smallheight + thy-1)/thy;

    switch (location){
        case (Vship_ChromaLoc_Left):
        case (Vship_ChromaLoc_TopLeft):
            if (subw == 0){
            } else if (subw == 1){
                bicubicHorizontalLeftUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp, src[1], width, smallheight);
                GPU_CHECK(hipGetLastError());
                bicubicHorizontalLeftUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp+width*smallheight, src[2], width, smallheight);
                GPU_CHECK(hipGetLastError());
            } else if (subw == 2){
                bicubicHorizontalLeftUpscaleX4_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp, src[1], width, smallheight);
                GPU_CHECK(hipGetLastError());
                bicubicHorizontalLeftUpscaleX4_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp+width*smallheight, src[2], width, smallheight);
                GPU_CHECK(hipGetLastError());
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Top):
            if (subw == 0){
            } else if (subw == 1){
                bicubicHorizontalCenterUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp, src[1], width, smallheight);
                GPU_CHECK(hipGetLastError());
                bicubicHorizontalCenterUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp+width*smallheight, src[2], width, smallheight);
                GPU_CHECK(hipGetLastError());
            } else if (subw == 2){
                bicubicHorizontalCenterUpscaleX4_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp, src[1], width, smallheight);
                GPU_CHECK(hipGetLastError());
                bicubicHorizontalCenterUpscaleX4_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(temp+width*smallheight, src[2], width, smallheight);
                GPU_CHECK(hipGetLastError());
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subw != 0) return 1; //not implemented
    }

    if (subh == 0){
        GPU_CHECK(hipMemcpyDtoDAsync(src[1], temp, sizeof(float)*width*smallheight, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(src[2], temp+width*smallheight, sizeof(float)*width*smallheight, stream));
    }
    //we need to copy the data to temp
    if (subw == 0){
        GPU_CHECK(hipMemcpyDtoDAsync(temp, src[1], sizeof(float)*width*smallheight, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(temp+width*smallheight, src[2], sizeof(float)*width*smallheight, stream));
    }

    blx = (width + thx-1)/thx;
    bly = (smallheight+1 + thy-1)/thy;

    switch (location){
        case (Vship_ChromaLoc_Top):
        case (Vship_ChromaLoc_TopLeft):
            if (subh == 0){
            } else if (subh == 1){
                bicubicVerticalTopUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(src[1], temp, width, height);
                GPU_CHECK(hipGetLastError());
                bicubicVerticalTopUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(src[2], temp+width*smallheight, width, height);
                GPU_CHECK(hipGetLastError());
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Left):
            if (subh == 0){
            } else if (subh == 1){
                bicubicVerticalCenterUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(src[1], temp, width, height);
                GPU_CHECK(hipGetLastError());
                bicubicVerticalCenterUpscaleX2_Kernel<<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(src[2], temp+width*smallheight, width, height);
                GPU_CHECK(hipGetLastError());
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subh != 0) return 1; //not implemented
    }

    return 0;
}

}