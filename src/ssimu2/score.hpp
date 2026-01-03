namespace ssimu2{

//if the weight is below or equal to weight_pruning, we skip and put to 0
const float weight_pruning = 0.01f;
const float weights[108] = {
    0.0f,
    0.0007376606707406586f,
    0.0f,
    0.0f,
    0.0007793481682867309f,
    0.0f,
    0.0f,
    0.0004371155730107379f,
    0.0f,
    1.1041726426657346f,
    0.00066284834129271f,
    0.00015231632783718752f,
    0.0f,
    0.0016406437456599754f,
    0.0f,
    1.8422455520539298f,
    11.441172603757666f,
    0.0f,
    0.0007989109436015163f,
    0.000176816438078653f,
    0.0f,
    1.8787594979546387f,
    10.94906990605142f,
    0.0f,
    0.0007289346991508072f,
    0.9677937080626833f,
    0.0f,
    0.00014003424285435884f,
    0.9981766977854967f,
    0.00031949755934435053f,
    0.0004550992113792063f,
    0.0f,
    0.0f,
    0.0013648766163243398f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    7.466890328078848f,
    0.0f,
    17.445833984131262f,
    0.0006235601634041466f,
    0.0f,
    0.0f,
    6.683678146179332f,
    0.00037724407979611296f,
    1.027889937768264f,
    225.20515300849274f,
    0.0f,
    0.0f,
    19.213238186143016f,
    0.0011401524586618361f,
    0.001237755635509985f,
    176.39317598450694f,
    0.0f,
    0.0f,
    24.43300999870476f,
    0.28520802612117757f,
    0.0004485436923833408f,
    0.0f,
    0.0f,
    0.0f,
    34.77906344483772f,
    44.835625328877896f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0008680556573291698f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0005313191874358747f,
    0.0f,
    0.00016533814161379112f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0004179171803251336f,
    0.0017290828234722833f,
    0.0f,
    0.0020827005846636437f,
    0.0f,
    0.0f,
    8.826982764996862f,
    23.19243343998926f,
    0.0f,
    95.1080498811086f,
    0.9863978034400682f,
    0.9834382792465353f,
    0.0012286405048278493f,
    171.2667255897307f,
    0.9807858872435379f,
    0.0f,
    0.0f,
    0.0f,
    0.0005130064588990679f,
    0.0f,
    0.00010854057858411537f,
};

struct SkipMap{
    bool ssim;
    bool artifact;
    bool detailloss;
};

SkipMap getSkipMap(int plane, int scale){
    SkipMap res;
    //we want both norms to be below pruning in order to avoid an error map computation
    res.ssim = weights[plane*6*2*3 + scale*2*3 + 0*3 + 0] <= weight_pruning;
    res.ssim &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 0] <= weight_pruning;
    res.artifact = weights[plane*6*2*3 + scale*2*3 + 0*3 + 1] <= weight_pruning;
    res.artifact &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 1] <= weight_pruning;
    res.detailloss = weights[plane*6*2*3 + scale*2*3 + 0*3 + 2] <= weight_pruning;
    res.detailloss &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 2] <= weight_pruning;
    return res;
}

int64_t tempAllocsizeScore(int64_t width, int64_t height){
    const int th_x = 16;
    const int th_y = 16;
    const int bl_x = (width-1)/th_x+1;
    const int bl_y = (height-1)/th_y+1;
    const int bl2_x = (width-1)/th_x/2+1;
    const int bl2_y = (height-1)/th_y/2+1;
    return 108 + 6*bl_x*bl_y + 6*bl2_x*bl2_y + 6*((bl_x*bl_y-1)/1024+1) + 6*((bl2_x*bl2_y-1)/1024+1);
}

//called by 1024 threads 
__global__ void sumreduce(float* dst, float* src, int widthPerElement){
    //dst must be of size 6*blocknum at least
    const int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    const int64_t thx = threadIdx.x;
    const int64_t threadnum = blockDim.x;
    
    __shared__ float sharedmem[6*1024];
    float* sum[6];
    #pragma unroll
    for (int i = 0; i < 6; i++){
        sum[i] = sharedmem + i*blockDim.x; //of size numthreads
    }

    if (x >= widthPerElement){
        #pragma unroll
        for (int i = 0; i < 6; i++){
            sum[i][thx] = 0;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 6; i++){
            sum[i][thx] = src[x + i*widthPerElement];
        }
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            #pragma unroll
            for (int i = 0; i < 6; i++){
                sum[i][thx] += sum[i][thx+next];
            }
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        #pragma unroll
        for (int i = 0; i < 6; i++){
            dst[i*gridDim.x + blockIdx.x] = sum[i][0];
        }
    }
}

//dst of size (6*grid_size_x*grid_size_y)
//th = dim3(16, 16)
template<bool skipSSIM, bool skipArtifact, bool skipDetailloss>
__global__ void planescale_map_Kernel(float* dst, float* im1, float* im2, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    const int64_t x = (threadIdx.x + blockIdx.x*16);
    const int64_t y = (threadIdx.y + blockIdx.y*16);

    const int64_t threadnum = 256;
    const int64_t thid = threadIdx.y*16 + threadIdx.x;

    constexpr uint gaussianBuffers = (skipSSIM) ? 2 : 3;
    __shared__ float sharedmem[32*32 * gaussianBuffers]; //12288B
    //explicit image cache
    float* im1tampon = sharedmem;
    float* im2tampon = sharedmem+32*32;
    //working buffer
    float* gaussiantampon;
    if constexpr (!skipSSIM){
        gaussiantampon = sharedmem+2*32*32;
    } else {
        (void)gaussiantampon;
    }

    //fill image buffers
    GaussianSmartSharedLoad(im1tampon, im1, x, y, width, height); 
    GaussianSmartSharedLoad(im2tampon, im2, x, y, width, height); 

    //our final values which may be unset depending on skip
    float im1p, im2p, m1, m2, s12, s11_s22;
    float ssim = 0, artifact = 0, detailloss = 0;

    //retrieve the value of im that we will use from the precreated tampon
    if constexpr (!skipArtifact || !skipDetailloss){
        im1p = im1tampon[(threadIdx.y+8)*32+threadIdx.x+8];
        im2p = im2tampon[(threadIdx.y+8)*32+threadIdx.x+8];
        __syncthreads();
    } else {
        //unused
        (void)im1p;
        (void)im2p;
    }

    if constexpr (!skipSSIM){
        //product blur a*b
        gaussiantampon[thid] = im1tampon[thid]*im2tampon[thid];
        gaussiantampon[thid+256] = im1tampon[thid+256]*im2tampon[thid+256];
        gaussiantampon[thid+512] = im1tampon[thid+512]*im2tampon[thid+512];
        gaussiantampon[thid+768] = im1tampon[thid+768]*im2tampon[thid+768];
        __syncthreads();
        s12 = GaussianSmart_Device(gaussiantampon, x, y, width, height, gaussiankernel, gaussiankernel_integral);

        //sum squared blur (a+b)**2
        float tmp;
        tmp = im1tampon[thid]+im2tampon[thid];
        gaussiantampon[thid] = tmp*tmp;
        tmp = im1tampon[thid+256]+im2tampon[thid+256];
        gaussiantampon[thid+256] = tmp*tmp;
        tmp = im1tampon[thid+512]+im2tampon[thid+512];
        gaussiantampon[thid+512] = tmp*tmp;
        tmp = im1tampon[thid+768]+im2tampon[thid+768];
        gaussiantampon[thid+768] = tmp*tmp;
        __syncthreads();
        const float sumsquared = GaussianSmart_Device(gaussiantampon, x, y, width, height, gaussiankernel, gaussiankernel_integral);
        //we can deduce a**2 + b**2 now following linearity of gaussian blur
        s11_s22 = sumsquared - 2*s12;
    } else {
        (void)s12; (void)s11_s22; //unused
    }

    //then we blur our im1buffer and im2buffer to get m1 and m2
    m1 = GaussianSmart_Device(im1tampon, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    m2 = GaussianSmart_Device(im2tampon, x, y, width, height, gaussiankernel, gaussiankernel_integral);

    //if (x == 0 && y == 0) printf("im1 %f im2 %f m1 %f m2 %f s12 %f s1122 %f\n", im1p, im2p, m1, m2, s12, s11_s22);
    //now we have im1p, im2p, m1, m2 and (s12, s11_s22 if ssim enabled)
    if (x < width && y < height){
        if constexpr (!skipSSIM){
            const float m11 = m1*m1;
            const float m22 = m2*m2;
            const float m12 = m1*m2;
            const float m_diff = m1-m2;
            const float num_m = fmaf(m_diff, m_diff*-1.0f, 1.0f);
            const float num_s = fmaf(s12 - m12, 2.0f, 0.0009f);
            const float denom_s = s11_s22 - m11 - m22 + 0.0009f;
            ssim = max(1.0f - ((num_m * num_s)/denom_s), 0.0f);
        }
        if constexpr (!skipArtifact || !skipDetailloss){
            //edge diff
            const float v1 = (abs(im2p - m2)+1.0f) / (abs(im1p - m1)+1.0f) - 1.0f;
            if constexpr (!skipArtifact) artifact = max(v1, 0.0f);
            if constexpr (!skipDetailloss) detailloss = max(v1*-1.0f, 0.0f);
        }
    }

    //accumulation of values across the block

    float* sumssim1 = sharedmem; //size sizeof(float)*threadnum
    float* sumssim4 = sharedmem+threadnum; //size sizeof(float)*threadnum
    float* suma1 = sharedmem+2*threadnum; //size sizeof(float)*threadnum
    float* suma4 = sharedmem+3*threadnum; //size sizeof(float)*threadnum
    float* sumd1 = sharedmem+4*threadnum; //size sizeof(float)*threadnum
    float* sumd4 = sharedmem+5*threadnum; //size sizeof(float)*threadnum

    sumssim1[thid] = ssim;
    sumssim4[thid] = tothe4th(ssim);
    suma1[thid] = artifact;
    suma4[thid] = tothe4th(artifact);
    sumd1[thid] = detailloss;
    sumd4[thid] = tothe4th(detailloss);
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thid + next < threadnum && (thid%(next*2) == 0)){
            sumssim1[thid] += sumssim1[thid+next];
            sumssim4[thid] += sumssim4[thid+next];
            suma1[thid] += suma1[thid+next];
            suma4[thid] += suma4[thid+next];
            sumd1[thid] += sumd1[thid+next];
            sumd4[thid] += sumd4[thid+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thid == 0){
        dst[blockIdx.y*gridDim.x + blockIdx.x] = sumssim1[0]/(width*height);
        dst[gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = suma1[0]/(width*height);
        dst[2*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = sumd1[0]/(width*height);
        dst[3*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = sumssim4[0]/(width*height);
        dst[4*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = suma4[0]/(width*height);
        dst[5*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x] = sumd4[0]/(width*height);
    }
}

void planescale_map_noaccumulate(float* dst, float* im1, float* im2, int64_t width, int64_t height, GaussianHandle& gaussianhandle, SkipMap skipMap, hipStream_t stream){
    const int th_x = 16;
    const int th_y = 16;
    const int bl_x = (width-1)/th_x+1;
    const int bl_y = (height-1)/th_y+1;
    if (skipMap.ssim){
        if (skipMap.artifact){
            if (skipMap.detailloss){
                return;
            } else {
                planescale_map_Kernel<true, true, false><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            }
        } else {
            if (skipMap.detailloss){
                planescale_map_Kernel<true, false, true><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            } else {
                planescale_map_Kernel<true, false, false><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            }
        }
    } else {
        if (skipMap.artifact){
            if (skipMap.detailloss){
                planescale_map_Kernel<false, true, true><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            } else {
                planescale_map_Kernel<false, true, false><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            }
        } else {
            if (skipMap.detailloss){
                planescale_map_Kernel<false, false, true><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            } else {
                planescale_map_Kernel<false, false, false><<<dim3(bl_x, bl_y), dim3(th_x, th_x), 0, stream>>>(dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
            }
        }
    }
    GPU_CHECK(hipGetLastError());
}

void planescale_map(float* outbuffer, float* im1, float* im2, float* temp, int64_t width, int64_t height, GaussianHandle& gaussianhandle, SkipMap skipMap, hipStream_t stream){
    //outbuffer is a device buffer of size 6 float which should get {map0norm1, map1norm1, map2norm1, map0norm4, map1norm4, map2norm4} at the end (still async to stream)
    //skipped values can remain unset
    if (skipMap.ssim && skipMap.artifact && skipMap.detailloss) return;

    int64_t firstbufferSize = ((width-1)/16+1)*((height-1)/16+1);
    float* tmps[2] = {temp, temp+6*firstbufferSize}; //of size firstbufferSize*6 and size 6*((firstbufferSize-1) / 1024+1)
    planescale_map_noaccumulate((firstbufferSize == 1) ? outbuffer:tmps[0], im1, im2, width, height, gaussianhandle, skipMap, stream);

    int oscillate = 0; //current valid buffer
    while (firstbufferSize != 1){
        sumreduce<<<dim3((firstbufferSize-1)/1024+1), dim3(1024), 0, stream>>>((firstbufferSize <= 1024) ? outbuffer:tmps[oscillate^1], tmps[oscillate], firstbufferSize);
        GPU_CHECK(hipGetLastError());

        oscillate ^= 1;
        firstbufferSize = (firstbufferSize-1)/1024+1;
    }
}

std::vector<float> allscore_map(float* im1[3], float* im2[3], float* temp, int64_t basewidth, int64_t baseheight, GaussianHandle& gaussianhandle, hipStream_t streams[2], hipEvent_t events[4]){
    //measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i]
    std::vector<float> res(108);
    float* outbuffer_d = temp;
    GPU_CHECK(hipMemsetAsync(outbuffer_d, 0, sizeof(float)*108, streams[0]));

    //we'll sync stream2 to stream1 and stream1 to stream2 in order to be sure each can manipule data from both sources
    GPU_CHECK(hipEventRecord(events[0], streams[0]));
    GPU_CHECK(hipStreamWaitEvent(streams[1], events[0]));
    GPU_CHECK(hipEventRecord(events[1], streams[1]));
    GPU_CHECK(hipStreamWaitEvent(streams[0], events[1]));

    float* temp1 = temp+108;
    const int th_x = 16;
    const int th_y = 16;
    const int bl_x = (basewidth-1)/th_x+1;
    const int bl_y = (baseheight-1)/th_y+1;
    float* temp2 = temp1+6*bl_x*bl_y + 6*((bl_x*bl_y-1)/1024+1);

    int64_t w = basewidth;
    int64_t h = baseheight;
    int64_t index = 0;
    //stream1 will exclusively manage scale 0 while stream2 will manage smaller scales
    for (int scale = 0; scale < 6; scale++){
        hipStream_t stream = (scale == 0) ? streams[0] : streams[1];
        float* tempScale = (scale == 0) ? temp1 : temp2;
        for (int plane = 0; plane < 3; plane++){
            SkipMap skipMap = getSkipMap(plane, scale);
            planescale_map(outbuffer_d+plane*6*2*3 + scale*2*3, im1[plane]+index, im2[plane]+index, tempScale, w, h, gaussianhandle, skipMap, stream);
        }
        index += w*h;
        w = (w+1)/2;
        h = (h+1)/2;
    }

    //stream 1 waits for stream 2 and then get the data back. Temp data is freed by stream 1 so when stream2 is here, it can free its data
    GPU_CHECK(hipEventRecord(events[2], streams[1]));
    GPU_CHECK(hipStreamWaitEvent(streams[0], events[2]));
    GPU_CHECK(hipMemcpyDtoHAsync(res.data(), outbuffer_d, sizeof(float)*108, streams[0]));
    GPU_CHECK(hipStreamSynchronize(streams[0]));
    //wait for stream 1 to be fully done which means both streams are done and the data is in res

    for (int scale = 0; scale < 6; scale++){
        for (int plane = 0; plane < 3; plane++){
            SkipMap skipMap = getSkipMap(plane, scale);
            if (!skipMap.ssim){
                //complete norm4
                res[plane*6*2*3 + scale*2*3 + 1*3 + 0] = std::sqrt(std::sqrt(res[plane*6*2*3 + scale*2*3 + 1*3 + 0]));
            }
            if (!skipMap.artifact){
                //complete norm4
                res[plane*6*2*3 + scale*2*3 + 1*3 + 1] = std::sqrt(std::sqrt(res[plane*6*2*3 + scale*2*3 + 1*3 + 1]));    
            }
            if (!skipMap.detailloss){
                //complete norm4
                res[plane*6*2*3 + scale*2*3 + 1*3 + 2] = std::sqrt(std::sqrt(res[plane*6*2*3 + scale*2*3 + 1*3 + 2]));
            }
        }
    }
    return res;
}

float final_score(std::vector<float> scores){
    //score has to be of size 108
    double ssim = 0.0f;
    for (int i = 0; i < 108; i++){
        ssim = fmaf(weights[i], scores[i], ssim);
    }
    ssim *= 0.9562382616834844;
    ssim = (6.248496625763138e-5 * ssim * ssim) * ssim +
        2.326765642916932 * ssim -
        0.020884521182843837 * ssim * ssim;
    
    if (ssim > 0.0) {
        ssim = std::pow(ssim, 0.6276336467831387) * -10.0 + 100.0;
    } else {
        ssim = 100.0f;
    }

    return ssim;
}

}