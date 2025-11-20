#pragma once

#include<iostream>

constexpr bool debug = true;

void writeGPUPlaneToFile(std::string filename, float* p, int64_t width, int64_t height, hipStream_t stream){
    std::ofstream f(filename);

    if (!f){
        std::cout << "Failed to open " << filename << " to write GPUPlane" << std::endl;
        return;
    }

    std::vector<float> cpu_p(width*height);
    hipMemcpyDtoHAsync(cpu_p.data(), p, sizeof(float)*width*height, stream);
    hipStreamSynchronize(stream);

    //write plane

    //metadata
    f << width << "," << height << std::endl;
    //data (written line then columns)
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            f << cpu_p[y*width+x] << ",";
        }
    }
    f << std::endl;
}