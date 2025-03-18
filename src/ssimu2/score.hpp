namespace ssimu2{

__global__ void sumreduce(float3* dst, float3* src, int bl_x){
    //dst must be of size 6*sizeof(float3)*blocknum at least
    //shared memory needed is 6*sizeof(float3)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    extern __shared__ float3 sharedmem[];
    float3* sumssim1 = sharedmem; //size sizeof(float3)*threadnum
    float3* sumssim4 = sharedmem+blockDim.x; //size sizeof(float3)*threadnum
    float3* suma1 = sharedmem+2*blockDim.x; //size sizeof(float3)*threadnum
    float3* suma4 = sharedmem+3*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd1 = sharedmem+4*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd4 = sharedmem+5*blockDim.x; //size sizeof(float3)*threadnum

    if (x >= bl_x){
        sumssim1[thx].x = 0.0f;
        sumssim4[thx].x = 0.0f;
        suma1[thx].x = 0.0f;
        suma4[thx].x = 0.0f;
        sumd1[thx].x = 0.0f;
        sumd4[thx].x = 0.0f;
        
        sumssim1[thx].y = 0.0f;
        sumssim4[thx].y = 0.0f;
        suma1[thx].y = 0.0f;
        suma4[thx].y = 0.0f;
        sumd1[thx].y = 0.0f;
        sumd4[thx].y = 0.0f;
        
        sumssim1[thx].z = 0.0f;
        sumssim4[thx].z = 0.0f;
        suma1[thx].z = 0.0f;
        suma4[thx].z = 0.0f;
        sumd1[thx].z = 0.0f;
        sumd4[thx].z = 0.0f;
    } else {
        sumssim1[thx] = src[x];
        sumssim4[thx] = src[x + bl_x];
        suma1[thx] = src[x + 2*bl_x];
        suma4[thx] = src[x + 3*bl_x];
        sumd1[thx] = src[x + 4*bl_x];
        sumd4[thx] = src[x + 5*bl_x];
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sumssim1[thx] += sumssim1[thx+next];
            sumssim4[thx] += sumssim4[thx+next];
            suma1[thx] += suma1[thx+next];
            suma4[thx] += suma4[thx+next];
            sumd1[thx] += sumd1[thx+next];
            sumd4[thx] += sumd4[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sumssim1[0];
        dst[gridDim.x + blockIdx.x] = sumssim4[0];
        dst[2*gridDim.x + blockIdx.x] = suma1[0];
        dst[3*gridDim.x + blockIdx.x] = suma4[0];
        dst[4*gridDim.x + blockIdx.x] = sumd1[0];
        dst[5*gridDim.x + blockIdx.x] = sumd4[0];
    }
}

__global__ void allscore_map_Kernel(float3* dst, float3* im1, float3* im2, float3* mu1, float3* mu2, float3* s11, float3* s22, float3* s12, int width, int height){
    //dst must be of size 6*sizeof(float3)*blocknum at least
    //shared memory needed is 6*sizeof(float3)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    extern __shared__ float3 sharedmem[];
    float3* sumssim1 = sharedmem; //size sizeof(float3)*threadnum
    float3* sumssim4 = sharedmem+blockDim.x; //size sizeof(float3)*threadnum
    float3* suma1 = sharedmem+2*blockDim.x; //size sizeof(float3)*threadnum
    float3* suma4 = sharedmem+3*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd1 = sharedmem+4*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd4 = sharedmem+5*blockDim.x; //size sizeof(float3)*threadnum
    

    float3 d0, d1, d2;
    if (x < width*height){
        //ssim
        const float3 m1 = mu1[x]; const float3 m2 = mu2[x];
        const float3 m11 = m1*m1;
        const float3 m22 = m2*m2;
        const float3 m12 = m1*m2;
        const float3 m_diff = m1-m2;
        const float3 num_m = fmaf(m_diff, m_diff*-1.0f, 1.0f);
        const float3 num_s = fmaf(s12[x] - m12, 2.0f, 0.0009f);
        const float3 denom_s = (s11[x] - m11) + (s22[x] - m22) + 0.0009f;
        d0 = max(1.0f - ((num_m * num_s)/denom_s), 0.0f);

        //edge diff
        const float3 v1 = (abs(im2[x] - mu2[x])+1.0f) / (abs(im1[x] - mu1[x])+1.0f) - 1.0f;
        const float3 artifact = max(v1, 0.0f);
        const float3 detailloss = max(v1*-1.0f, 0.0f);
        d1 = artifact; d2 = detailloss;
    } else {
        d0.x = 0.0f;
        d0.y = 0.0f;
        d0.z = 0.0f;
        d1.x = 0.0f;
        d1.y = 0.0f;
        d1.z = 0.0f;
        d2.x = 0.0f;
        d2.y = 0.0f;
        d2.z = 0.0f;
    }

    sumssim1[thx] = d0;
    sumssim4[thx] = tothe4th(d0);
    suma1[thx] = d1;
    suma4[thx] = tothe4th(d1);
    sumd1[thx] = d2;
    sumd4[thx] = tothe4th(d2);
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sumssim1[thx] += sumssim1[thx+next];
            sumssim4[thx] += sumssim4[thx+next];
            suma1[thx] += suma1[thx+next];
            suma4[thx] += suma4[thx+next];
            sumd1[thx] += sumd1[thx+next];
            sumd4[thx] += sumd4[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sumssim1[0]/(width*height);
        dst[gridDim.x + blockIdx.x] = sumssim4[0]/(width*height);
        dst[2*gridDim.x + blockIdx.x] = suma1[0]/(width*height);
        dst[3*gridDim.x + blockIdx.x] = suma4[0]/(width*height);
        dst[4*gridDim.x + blockIdx.x] = sumd1[0]/(width*height);
        dst[5*gridDim.x + blockIdx.x] = sumd4[0]/(width*height);
    }
}

std::vector<float3> allscore_map(float3* im1, float3* im2, float3* mu1, float3* mu2, float3* s11, float3* s22, float3* s12, float3* temp, int basewidth, int baseheight, int maxshared, hipEvent_t event_d, hipStream_t stream){
    //output is {normssim1scale1, normssim4scale1, norma1scale1, norma4scale1, normad1scale1, normd4scale1, norm1scale2, ...}
    std::vector<float3> result(2*6*3);
    for (int i = 0; i < 2*6*3; i++) {result[i].x = 0.0f; result[i].y = 0.0f; result[i].z = 0.0f;}

    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    std::vector<int> scaleoutdone(7);
    scaleoutdone[0] = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min((int)(maxshared/(6*sizeof(float3)))/32*32, std::min(1024, w*h));
        bl_x = (w*h-1)/th_x + 1;
        int blr_x = bl_x;
        allscore_map_Kernel<<<dim3(bl_x), dim3(th_x), 6*sizeof(float3)*th_x, stream>>>(temp+scaleoutdone[scale]+((blr_x >= th_x) ? 6*bl_x : 0), im1+index, im2+index, mu1+index, mu2+index, s11+index, s22+index, s12+index, w, h);
        //printf("I got %s with %d\n", hipGetErrorString(hipGetLastError()), 6*sizeof(float3)*th_x);
        GPU_CHECK(hipGetLastError());

        int oscillate = 0; //3 sets of memory: real destination at 0, first at 6*bl_x for oscillate 0 and last at 12*bl_x for oscillate 1;
        int oldblr_x = bl_x;
        while (blr_x >= th_x){
            blr_x = (bl_x -1)/th_x + 1;
            //sum reduce
            sumreduce<<<dim3(blr_x), dim3(th_x), 6*sizeof(float3)*th_x, stream>>>(temp+scaleoutdone[scale]+((blr_x >= th_x) ? (oscillate+1)*6*bl_x : 0), temp+scaleoutdone[scale]+ (oscillate+1)*6*bl_x, oldblr_x);
            oscillate ^= 1;
            oldblr_x = blr_x;
        }

        scaleoutdone[scale+1] = scaleoutdone[scale]+6*blr_x;
        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    float3* hostback = (float3*)malloc(sizeof(float3)*scaleoutdone[6]);
    if (!hostback) throw std::bad_alloc();
    GPU_CHECK(hipMemcpyDtoHAsync(hostback, (hipDeviceptr_t)temp, sizeof(float3)*scaleoutdone[6], stream));
    //the data as already been reduced by a factor of 512 which can now be reasonably retrieved from GPU

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d);

    //let s reduce!
    for (int scale = 0; scale < 6; scale++){
        bl_x = (scaleoutdone[scale+1] - scaleoutdone[scale])/6;
        for (int i = 0; i < 6*bl_x; i++){
            if (i < bl_x){
                result[6*scale] += hostback[scaleoutdone[scale] + i];
            } else if (i < 2*bl_x) {
                result[6*scale+1] += hostback[scaleoutdone[scale] + i];
            } else if (i < 3*bl_x) {
                result[6*scale+2] += hostback[scaleoutdone[scale] + i];
            } else if (i < 4*bl_x) {
                result[6*scale+3] += hostback[scaleoutdone[scale] + i];
            } else if (i < 5*bl_x) {
                result[6*scale+4] += hostback[scaleoutdone[scale] + i];
            } else {
                result[6*scale+5] += hostback[scaleoutdone[scale] + i];
            }
        }
    }

    free(hostback);

    for (int i = 0; i < 18; i++){
        result[2*i+1].x = std::sqrtf(std::sqrtf(result[2*i+1].x));
        result[2*i+1].y = std::sqrtf(std::sqrtf(result[2*i+1].y));
        result[2*i+1].z = std::sqrtf(std::sqrtf(result[2*i+1].z));
    } //completing 4th norm

    return result;
}

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

float final_score(std::vector<float> scores){
    //score has to be of size 108
    float ssim = 0.0f;
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