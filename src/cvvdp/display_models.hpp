#pragma once

namespace cvvdp{

enum Colorspace{
    sRGB,
    BT709LINEAR,
    BT2020HLG,
    BT2020PQ,
};

struct DisplayModel{
    std::string name = "";
    int64_t resolution[2] = {1920, 1080};
    Colorspace colorspace = sRGB;
    float viewing_distance_meters;
    float diagonal_size_inches;
    float max_luminance;
    float contrast = 500;
    float E_ambient = 0;
    float k_refl = 0.005;
    float exposure = 1;
    std::string source = "";

    //computed
    float cached_ppd = -1;
    DisplayModel(std::string model_key){
        if (model_key == "standard_4k"){
            name = "30-inch 4K monitor, peak luminance 200 cd/m^2, viewed under office light levels (250 lux), seen from 2 x display height";
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 200;
            contrast = 1000;
            E_ambient = 250;
            source = "none";
        } else if (model_key == "standard_hdr_pq"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed under low light levels (10 lux), seen from 2 x display height";
            colorspace = BT2020PQ;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_hdr_hlg"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed under low light levels (10 lux), seen from 2 x display height";
            colorspace = BT2020HLG;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_hdr_linear"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed under low light levels (10 lux), seen from 2 x display height";
            colorspace = BT709LINEAR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_hdr_dark"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed in a dark room (0 lux), seen from 2 x display height";
            colorspace = BT709LINEAR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 0;
            source = "none";
        } else if (model_key == "standard_hdr_linear_zoom"){
            name = "30-inch 4K HDR monitor, peak luminance 4000 cd/m^2, viewed under low light levels (10 lux), seen from very close to spot super-resolution artifacts";
            colorspace = BT709LINEAR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.25;
            diagonal_size_inches = 30;
            max_luminance = 10000;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_fhd"){
            name = "24-inch FullHD monitor, peak luminance 200 cd/m^2, viewed under office light levels (250 lux), seen from 2 x display height";
            resolution[0] = 1920; resolution[1] = 1080;
            viewing_distance_meters = 0.6;
            diagonal_size_inches = 24;
            max_luminance = 200;
            contrast = 1000;
            E_ambient = 250;
            source = "none";
        } else {
            throw VshipError(BadDisplayModel, __FILE__, __LINE__);
        }
    }
    float get_screen_ppd(){
        if (cached_ppd != -1) return cached_ppd;

        const float ar = (float)resolution[0]/(float)resolution[1];
        const float height_meter = std::sqrt((diagonal_size_inches*25.4)*(diagonal_size_inches*25.4) / (1 + ar*ar))/1000;
        const float width_meter = ar*height_meter;

        const float pix_deg = 2*180*std::atan(0.5*width_meter/(float)resolution[0]/viewing_distance_meters)/PI;
        cached_ppd = 1/pix_deg;
        return cached_ppd;
    }
    float getBlackLevel(){
        return E_ambient*k_refl/PI;
    }
    float getReflLevel(){
        return max_luminance/contrast;
    }
};

}