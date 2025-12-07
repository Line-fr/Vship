#pragma once

#include<fstream>
#include<ios>
#include<cctype>

namespace cvvdp{

enum DisplayColorspace{
    sRGB,
    HDR,
};

struct DisplayModel{
    std::string name = "";
    int64_t resolution[2] = {1920, 1080};
    DisplayColorspace colorspace = sRGB;
    float viewing_distance_meters = 0.6;
    float diagonal_size_inches = 24;
    float max_luminance = 200;
    float contrast = 1000;
    float E_ambient = 250;
    float k_refl = 0.005;
    float exposure = 1;
    std::string source = "";

    bool resolutionSet = true;
    bool colorspaceSet = true;
    bool viewing_distance_metersSet = true;
    bool diagonal_size_inchesSet = true;
    bool max_luminanceSet = true;
    bool contrastSet = true;
    bool E_ambientSet = true;
    bool k_reflSet = true;
    bool exposureSet = true;

    //computed
    float cached_ppd = -1;
    DisplayModel(std::string model_key, std::string model_config_json){
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
            colorspace = HDR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        //same as PQ, what matters is that it is HDR
        } else if (model_key == "standard_hdr_hlg"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed under low light levels (10 lux), seen from 2 x display height";
            colorspace = HDR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_hdr_linear"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed under low light levels (10 lux), seen from 2 x display height";
            colorspace = HDR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 10;
            source = "none";
        } else if (model_key == "standard_hdr_dark"){
            name = "30-inch 4K HDR monitor, peak luminance 1500 cd/m^2, viewed in a dark room (0 lux), seen from 2 x display height";
            colorspace = HDR;
            resolution[0] = 3840; resolution[1] = 2160;
            viewing_distance_meters = 0.7472;
            diagonal_size_inches = 30;
            max_luminance = 1500;
            contrast = 1000000;
            E_ambient = 0;
            source = "none";
        } else if (model_key == "standard_hdr_linear_zoom"){
            name = "30-inch 4K HDR monitor, peak luminance 4000 cd/m^2, viewed under low light levels (10 lux), seen from very close to spot super-resolution artifacts";
            colorspace = HDR;
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
            resolutionSet = false;
            colorspaceSet = false;
            viewing_distance_metersSet = false;
            diagonal_size_inchesSet = false;
            max_luminanceSet = false;
            contrastSet = false;
            E_ambientSet = false;
            k_reflSet = false;
            exposureSet = false;
        }

        //auto timeinit = std::chrono::high_resolution_clock::now();
        int jsonret = 1;
        if (model_config_json != ""){
            jsonret = parseJson(model_config_json, model_key);
        }
        //auto timeend = std::chrono::high_resolution_clock::now();
        //const uint64_t milli = std::chrono::duration_cast<std::chrono::microseconds>(timeend - timeinit).count();
        //std::cout << "time for parsing: " << milli << " micro seconds" << std::endl;

        std::string detail = "Display Unset, wrong display name?";
        bool errorOut = false;
        //first case is: model name not in json nor in standard
        if (jsonret == 1 && resolutionSet == false){errorOut = true;}
        else if (!resolutionSet){errorOut = true; detail = "Display Missing resolution";}
        else if (!colorspaceSet){errorOut = true; detail = "Display Missing colorspace";}
        else if (!viewing_distance_metersSet){errorOut = true; detail = "Display Missing viewing_distance_meters";}
        else if (!diagonal_size_inchesSet){errorOut = true; detail = "Display Missing diagonal_size_inches";}
        else if (!max_luminanceSet){errorOut = true; detail = "Display Missing max_luminance";}
        else if (!contrastSet){errorOut = true; detail = "Display Missing contrast";}
        else if (!E_ambientSet){errorOut = true; detail = "Display Missing E_ambiant";}
        //else if (!k_reflSet){errorOut = true; detail = "Display Missing k_refl";}
        //else if (!exposureSet){errorOut = true; detail = "Display Missing exposure";}
        if (errorOut) throw VshipError(BadDisplayModel, __FILE__, __LINE__, detail);
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
    float getReflLevel(){
        return E_ambient*k_refl/PI;
    }
    float getBlackLevel(){
        return max_luminance/contrast;
    }
private:
    //returns 0 if the model is set (ie everything is fine and the model name is found)
    //returns 1 if the model isnt set (ie everything is fine and model name is not found inside)
    //throws exception otherwise
    int parseJson(std::string jsonfile, std::string model_name){
        std::ifstream file(jsonfile);

        if (!file) throw VshipError(BadPath, __FILE__, __LINE__, "Path: "+jsonfile);
        
        //auto timeinit = std::chrono::high_resolution_clock::now();

        int ret = parseMasterDictionary(file, model_name);
        
        //auto timeend = std::chrono::high_resolution_clock::now();
        //const uint64_t milli = std::chrono::duration_cast<std::chrono::microseconds>(timeend - timeinit).count();
        //std::cout << "time for parsing: " << milli << " micro seconds" << std::endl;
        
        return ret;
    }
    int parseMasterDictionary(std::ifstream& stream, std::string& model_name){
        int ret = 1;

        //get inside main dictionnary
        passUntil<'{'>(stream, true);
        //get key
        while (passUntil2<'"', '}'>(stream, true) == 0){
            //we are at a key definition point
            std::string key = parseString(stream);
            passUntil<':'>(stream, false);
            passUntil<'{'>(stream, false); //open the dic
            if (key != model_name) {
                //we pass the model since it is not the one we want
                closeCurrentDic(stream); 
                //close it no matter what is inside to be ready for next key
            } else {
                //we have the model we wish
                parseDisplayDictionary(stream);
                ret = 0;
            }
            if (passUntil2<',', '}'>(stream, true) == 0){
                continue; //there is a key behind (or an end of dic)
            } else {
                break; //directly the end
            }
        }

        return ret;
    }
    //we have passed the opening { and will need to parse the final } before stopping
    void parseDisplayDictionary(std::ifstream& stream){
        while (passUntil2<'"', '}'>(stream, true) == 0){
            //we are at a key definition point
            std::string key = parseString(stream);
            passUntil<':'>(stream, false);
            if (key == "name"){
                passUntil<'"'>(stream, false);
                name = parseString(stream);
            } else if (key == "source"){
                passUntil<'"'>(stream, false);
                source = parseString(stream);
            } else if (key == "colorspace") {
                passUntil<'"'>(stream, false);
                std::string theircolorspace = parseString(stream);
                if (theircolorspace == "HDR" || theircolorspace == "BT.2020-PQ" || theircolorspace == "BT.2020-HLG"){
                    colorspace = HDR;
                } else if (theircolorspace == "BT.709-linear" || theircolorspace == "sRGB" || theircolorspace == "SDR"){
                    colorspace = sRGB;
                } else {
                    throw VshipError(BadJson, __FILE__, __LINE__, "colospace can either be BT.2020-PQ, BT.2020-HLG, BT.709-linear, HDR, sRGB or SDR");
                }
                colorspaceSet = true;
            } else if (key == "resolution"){
                //list of int
                passUntil<'['>(stream, false);
                resolution[0] = parseFloat(stream, true);
                passUntil<','>(stream, true);
                resolution[1] = parseFloat(stream, true);
                passUntil<']'>(stream, true);
                resolutionSet = true;
                //std::cout << "resolution " << resolution[0] << "x" << resolution[1] << std::endl;
            } else if (key == "viewing_distance_meters"){
                viewing_distance_meters = parseFloat(stream, true);
                viewing_distance_metersSet = true;
            } else if (key == "diagonal_size_inches"){
                diagonal_size_inches = parseFloat(stream, true);
                diagonal_size_inchesSet = true;
            } else if (key == "max_luminance"){
                max_luminance = parseFloat(stream, true);
                max_luminanceSet = true;
            } else if (key == "contrast"){
                contrast = parseFloat(stream, true);
                contrastSet = true;
            } else if (key == "E_ambient"){
                E_ambient = parseFloat(stream, true);
                E_ambientSet = true;
            } else if (key == "k_refl"){
                k_refl = parseFloat(stream, true);
                k_reflSet = true;
            } else if (key == "exposure"){
                exposure = parseFloat(stream, true);
                exposureSet = true;
            } else {
                throw VshipError(BadJson, __FILE__, __LINE__, "Unsupported field "+key+" in selected display");
            }

            if (passUntil2<',', '}'>(stream, true) == 0){
                continue; //there is a key behind (or an end of dic)
            } else {
                break; //directly the end
            }
        }
    }
    //exception if fail
    //return the element that got trapped
    template<char ch>
    int passUntil(std::ifstream& stream, const bool allowLineRet){
        char t;
        std::string moredetail = "";
        while (moredetail == "" && stream >> std::noskipws >> t){
            switch (t){
                case ch:
                    return 0;
                case '\n':
                    if (!allowLineRet) {
                        moredetail = "Illformed json, line ret where not allowed here.";
                    }
                case '\t':
                case ' ':
                    continue;
                default:
                    moredetail = "Illformed json, found ";
                    moredetail.push_back(t);
                    continue;
            }
            
        }
        moredetail += " missing character `";
        moredetail.push_back(ch);
        moredetail += "`";
        throw VshipError(BadJson, __FILE__, __LINE__, moredetail);
    }
    template<char ch, char ch2>
    int passUntil2(std::ifstream& stream, const bool allowLineRet){
        char t;
        std::string moredetail = "";
        while (moredetail == "" && stream >> std::noskipws >> t){
            switch (t){
                case ch:
                    return 0;
                case ch2:
                    return 1;
                case '\n':
                    if (!allowLineRet) {
                        moredetail = "Illformed json, line ret where not allowed here.";
                    }
                case '\t':
                case ' ':
                    continue;
                default:
                    moredetail = "Illformed json, found ";
                    moredetail.push_back(t);
                    continue;
            }
            
        }
        moredetail += " missing character `";
        moredetail.push_back(ch);
        moredetail += "` or `";
        moredetail.push_back(ch2);
        moredetail += "`";

        throw VshipError(BadJson, __FILE__, __LINE__, moredetail);
    }
    //stops when there is one more } than {
    void closeCurrentDic(std::ifstream& stream){
        char t;
        int counter = 1;
        bool isinstring = false;
        while (stream >> std::skipws >> t){
            if (isinstring){
                if (t == '"'){
                    isinstring = false;
                } else if (t == '\\'){
                    throw VshipError(BadJson, __FILE__, __LINE__, "No \\ accepted in strings");
                }
            } else {
                if (t == '{') {
                    counter++;
                } else if (t == '}'){
                    counter--;
                    if (counter == 0) return;
                } else if (t == '"'){
                    isinstring = true;
                }
            }
        }
        throw VshipError(BadJson, __FILE__, __LINE__, "Unclosed {");
    }
    //doesnt accept '\', stops directly at a "
    std::string parseString(std::ifstream& stream){
        std::string res = "";
        char t;
        while (stream >> std::noskipws >> t){
            if (t == '\\') {
                throw VshipError(BadJson, __FILE__, __LINE__, "No \\ accepted in strings");
            } else if (t == '"'){
                return res;
            } else {
                res.push_back(t);
            }
        }
        throw VshipError(BadJson, __FILE__, __LINE__, "String is not terminated at EOF");
        return "";
    }
    float parseFloat(std::ifstream& stream, bool allowLineRet){
        char t;
        //skipping until digit is found
        do{
            int c = stream.peek();
            if (c == EOF){
                throw VshipError(BadJson, __FILE__, __LINE__, "Illformed json, tried to find a float and found EOF");
            } else if (c == '\n'){
                if (!allowLineRet) {
                    throw VshipError(BadJson, __FILE__, __LINE__, "Illformed json, line ret where not allowed here. (searching for float)");
                }
                continue;
            } else if (c == '\t'){
                continue;
            } else if (c == ' '){
                continue;
            } else {
                if (std::isdigit(c) || c == '.'){
                    break;
                } else {
                    throw VshipError(BadJson, __FILE__, __LINE__, "Illformed json, non digit character found while searching for float");
                }
            }
        } while (stream >> t);

        //now we can parse a normal float knowing first digit is in res
        float res;
        stream >> res;
        return res;
    }
};

}