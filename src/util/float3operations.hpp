#ifndef FLOAT3OPHPP
#define FLOAT3OPHPP

__device__ __host__ void inline operator/=(float3& a, const float3& b){
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__device__ __host__ void inline operator/=(float3& a, const float b){
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
__device__ __host__ float3 inline operator/(const float3&a, const int b){
    float3 out;
    out.x = a.x/b;
    out.y = a.y/b;
    out.z = a.z/b;
    return out;
}

__device__ __host__ float3 inline operator-(const float a, const float3& b){
    float3 out;
    out.x = a - b.x;
    out.y = a - b.y;
    out.z = a - b.z;
    return out;
}

__device__ __host__ float3 inline operator-(const float3& a, const float b){
    float3 out;
    out.x = a.x - b;
    out.y = a.y - b;
    out.z = a.z - b;
    return out;
}

__device__ __host__ float3 inline max(const float3& a, const float& b){
    float3 out;
    out.x = max(a.x, b);
    out.y = max(a.y, b);
    out.z = max(a.z, b);
    return out;
}
__device__ __host__ float3 inline operator+(const float3& a, const float& b){
    float3 out;
    out.x = a.x + b;
    out.y = a.y + b;
    out.z = a.z + b;
    return out;
}

__device__ __host__ float3 inline fmaf(const float3& a, const float3& b, const float& c){
    float3 out;
    out.x = fmaf(a.x, b.x, c);
    out.y = fmaf(a.y, b.y, c);
    out.z = fmaf(a.z, b.z, c);
    return out;
}
__device__ __host__ float3 inline fmaf(const float3& a, const float& b, const float& c){
    float3 out;
    out.x = fmaf(a.x, b, c);
    out.y = fmaf(a.y, b, c);
    out.z = fmaf(a.z, b, c);
    return out;
}
__device__ __host__ float3 inline abs(const float3& a){
    float3 out;
    out.x = fabsf(a.x);
    out.y = fabsf(a.y);
    out.z = fabsf(a.z);
    return out;
}

#ifndef __HIPCC__

__device__ __host__ void inline operator*=(float3& a, const float3& b){
   a.x *= b.x;
   a.y *= b.y;
   a.z *= b.z;
}

__device__ __host__ void inline operator*=(float3& a, const float& b){
   a.x *= b;
   a.y *= b;
   a.z *= b;
}

__device__ __host__ void inline operator+=(float3& a, const float3& b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__device__ __host__ float3 inline operator*(const float3& a, const float3& b){
    float3 out;
    out.x = a.x*b.x;
    out.y = a.y*b.y;
    out.z = a.z*b.z;
    return out;
}
__device__ __host__ float3 inline operator/(const float3& a, const float3& b){
    float3 out;
    out.x = a.x/b.x;
    out.y = a.y/b.y;
    out.z = a.z/b.z;
    return out;
}
__device__ __host__ float3 inline operator-(const float3& a, const float3& b){
    float3 out;
    out.x = a.x - b.x;
    out.y = a.y - b.y;
    out.z = a.z - b.z;
    return out;
}
__device__ __host__ float3 inline operator+(const float3& a, const float3& b){
    float3 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    return out;
}
__device__ __host__ float3 inline fmaf(const float3& a, const float3& b, const float3& c){
    float3 out;
    out.x = fmaf(a.x, b.x, c.x);
    out.y = fmaf(a.y, b.y, c.y);
    out.z = fmaf(a.z, b.z, c.z);
    return out;
}
#endif

__device__ __host__ float3 inline operator*(const float3& a, const float& b){
    float3 out;
    out.x = a.x*b;
    out.y = a.y*b;
    out.z = a.z*b;
    return out;
}
__device__ __host__ float tothe4th(float x){
    float y = x*x;
    return y*y;
}

__device__ __host__ float3 tothe4th(float3 x){
    float3 y = x*x;
    return y*y;
}

__launch_bounds__(256)
__global__ void multarray_Kernel(float3* src1, float3* src2, float3* dst, int width){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    dst[x] = src1[x]*src2[x];
}

void multarray(float3* src1, float3* src2, float3* dst, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    multarray_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void subarray_Kernel(float* src1, float* src2, float* dst, int width){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    dst[x] = src1[x]-src2[x];
}

void subarray(float* src1, float* src2, float* dst, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    subarray_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void memoryorganizer_kernel(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    out[i*width + j].x = ((float*)(srcp0 + i*stride))[j];
    out[i*width + j].y = ((float*)(srcp1 + i*stride))[j];
    out[i*width + j].z = ((float*)(srcp2 + i*stride))[j];
}

void memoryorganizer(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    memoryorganizer_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(out, srcp0, srcp1, srcp2, stride, width, height);
}

__launch_bounds__(256)
__global__ void strideEliminator_kernel(float* mem_d, const uint8_t* src, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    mem_d[i*width+j] = ((float*)(src + i*stride))[j];
}

__launch_bounds__(256)
__global__ void strideAdder_kernel(const uint8_t* dst, float* mem_d, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    ((float*)(dst + i*stride))[j] = mem_d[i*width+j];
}

#endif