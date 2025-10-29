#pragma once

namespace cvvdp{

constexpr float mask_p = 2.264355182647705;
constexpr float mask_c = -0.7954971194267273;
constexpr int pu_dilate = 3;
constexpr int beta = 2;
constexpr int beta_t = 2;
constexpr int beta_tch = 4;
constexpr int beta_sch = 4;
constexpr float csf_sigma = -1.5;
constexpr float sensitivity_correction = -0.2797423303127289;
constexpr float jod_a = 0.0439569391310215;
constexpr float jod_exp = 0.9302042722702026;
__device__ constexpr float mask_q[4] = {
    1.302622675895691,
    2.8885908126831055,
    3.6807713508605957,
    3.588787317276001,
};
constexpr int filter_len = -1;
constexpr float ch_chrom_w = 1.;
constexpr float ch_trans_w = 0.8081134557723999;
constexpr float sigma_tf[4] = {
    5.79336,
    14.1255,
    6.63661,
    0.12314,
};
constexpr float beta_tf[4] = {
    1.3314,
    1.1196,
    0.947901,
    0.1898
};
//constexpr bool xchannel_masking = true;
__device__ constexpr float xcm_weights[16] = {
    -0.18950104713439941,
    -5.962151050567627,
    -4.31834602355957,
    -1.9321587085723877,
    2.5655593872070312,
    0.34406712651252747,
    -2.719646453857422,
    -0.4970424771308899,
    3.8118371963500977,
    -1.0051705837249756,
    -0.5193376541137695,
    -0.5653647780418396,
    -7.054771423339844,
    -5.527150630950928,
    -3.5106418132781982,
    -2.08804988861084,
};
__device__ constexpr float baseband_weight[4] = {
    0.0036334486212581396,
    1.6627724170684814,
    4.11874532699585,
    25.25969886779785
};
constexpr float d_max = 2.5642454624176025;
constexpr float image_int = 0.577918291091919;
constexpr float bfilt_duration = 0.4;

}