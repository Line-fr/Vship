#pragma once

namespace VshipColorConvert{

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
};

__device__ CubicHermitSplineInterpolator getHorizontalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height-1);
    x = min(x, width); //extra space beware

    const float elm1 = (x == 0) ? src[y*width] : src[y*width+x-1];
    const float el0 = (x == width) ? src[y*width+width-1] : src[y*width+x];
    const float el1 = (x == width-1) ? el0 : src[y*width+x+1];
    const float el2 = (x == width-1) ? el1 : src[y*width+x+2];

    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

__device__ CubicHermitSplineInterpolator getVerticalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height);
    x = min(x, width-1);

    const float elm1 = (y == 0) ? src[x] : src[(y-1)*width+x];
    const float el0 = (y == height) ? src[(height-1)*width + x] : src[y*height+x];
    const float el1 = (y == height-1) ? el0 : src[(y+1)*height+x];
    const float el2 = (y == height-2) ? el1 : src[(y+2)*height+x];

    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

//block x should range from 0 to width INCLUDED
//dst of size (2*width)*height while src is of size width*height
__global__ void bicubicHorizontalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x -1;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height && x < width){
        if (x != -1) dst[y*2*width + 2*x+1] = interpolator.get(0.25);
        if (x != width-1) dst[y*2*width + 2*x+2] = interpolator.get(0.75);
    }
}

//block x should range from 0 to width-1 INCLUDED
//dst of size (2*width)*height while src is of size width*height
__global__ void bicubicHorizontalLeftUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0 and 0.5 (0 is directly our value)
    if (y < height && x < width){
        dst[y*2*width + 2*x] = src[y*width + x];
        dst[y*2*width + 2*x+1] = interpolator.get(0.5);
    }
}

//block x should range from 0 to width INCLUDED
//dst of size (4*width)*height while src is of size width*height
__global__ void bicubicHorizontalCenterUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x -1;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.125, 0.375, 0.625 and 0.875
    if (y < height && x < width){
        if (x != -1) dst[y*4*width + 4*x+2] = interpolator.get(0.125);
        if (x != -1) dst[y*4*width + 4*x+3] = interpolator.get(0.375);
        if (x != width-1) dst[y*4*width + 4*x+4] = interpolator.get(0.625);
        if (x != width-1) dst[y*4*width + 4*x+5] = interpolator.get(0.875);
    }
}

//block x should range from 0 to width-1 INCLUDED
//dst of size (4*width)*height while src is of size width*height
__global__ void bicubicHorizontalLeftUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0, 0.25, 0.5 and 0.75 (0 is directly our value)
    if (y < height && x < width){
        dst[y*4*width + 4*x] = src[y*width + x];
        dst[y*4*width + 4*x+1] = interpolator.get(0.25);
        dst[y*4*width + 4*x+2] = interpolator.get(0.5);
        dst[y*4*width + 4*x+3] = interpolator.get(0.75);
    }
}

//block y should range from 0 to width INCLUDED
//dst of size width*(2*height) while src is of size width*height
__global__ void bicubicVerticalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y -1;
    CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height && x < width){
        if (y != -1) dst[(2*y +1)*width + x] = interpolator.get(0.25);
        if (y != height-1) dst[(2*y+2)*width + x] = interpolator.get(0.75);
    }
}

//block y should range from 0 to width-1 INCLUDED
//dst of size width*(2*height) while src is of size width*height
__global__ void bicubicVerticalTopUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Top so we are interested in values: 0 and 0.5
    if (y < height && x < width){
        dst[(2*y)*width + x] = src[y*width + x];
        dst[(2*y+1)*width + x] = interpolator.get(0.5);
    }
}

//source is of size width * height possibly chroma downsampled
__host__ int inline upsample(float* dst[3], float* src[3], int64_t width, int64_t height, Vship_ChromaLocation_t location, Vship_ChromaSubsample_t subsampledata, hipStream_t stream){
    const int subw = subsampledata.subw;
    const int subh = subsampledata.subh;
    
    if (subw == 0 && subh == 0) return 0;
    width >>= subw; //get chroma plane size
    height >>= subh; 
    const int thx = 16;
    const int thy = 16;
    const int blx1 = (width + thx-1)/thx;
    const int blx2 = (width+1 + thx-1)/thx;
    const int bly1 = (height + thy-1)/thy;
    const int bly2 = (height+1 + thy-1)/thy;

    switch (location){
        case (Vship_ChromaLoc_Left):
        case (Vship_ChromaLoc_TopLeft):
            if (subw == 0){
            } else if (subw == 1){
                bicubicHorizontalLeftUpscaleX2_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicHorizontalLeftUpscaleX2_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                width *= 2;
            } else if (subw == 2){
                bicubicHorizontalLeftUpscaleX4_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicHorizontalLeftUpscaleX4_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                width *= 4;
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Top):
            if (subw == 0){
            } else if (subw == 1){
                bicubicHorizontalCenterUpscaleX2_Kernel<<<dim3(blx2, bly1), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicHorizontalCenterUpscaleX2_Kernel<<<dim3(blx2, bly1), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                width *= 2;
            } else if (subw == 2){
                bicubicHorizontalCenterUpscaleX4_Kernel<<<dim3(blx2, bly1), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicHorizontalCenterUpscaleX4_Kernel<<<dim3(blx2, bly1), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                width *= 4;
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subw != 0) return 1; //not implemented
    }

    switch (location){
        case (Vship_ChromaLoc_Top):
        case (Vship_ChromaLoc_TopLeft):
            if (subh == 0){
            } else if (subh == 1){
                bicubicVerticalTopUpscaleX2_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicVerticalTopUpscaleX2_Kernel<<<dim3(blx1, bly1), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                height *= 2;
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Left):
            if (subh == 0){
            } else if (subh == 1){
                bicubicVerticalCenterUpscaleX2_Kernel<<<dim3(blx1, bly2), dim3(thx, thy), 0, stream>>>(dst[1], src[1], width, height);
                bicubicVerticalCenterUpscaleX2_Kernel<<<dim3(blx1, bly2), dim3(thx, thy), 0, stream>>>(dst[2], src[2], width, height);
                height *= 2;
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