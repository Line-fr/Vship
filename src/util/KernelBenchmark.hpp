#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "VshipExceptions.hpp"

namespace helper{

constexpr bool libvship_bench = false;
const std::string libvship_benchJson = "libvshipBench.json";

class HIPTimerManager{
private:
    std::vector<hipEvent_t> events;
    std::vector<std::string> labels;
public:
    ~HIPTimerManager(){
        for (const auto& event: events){
            hipEventDestroy(event);
        }
    }
    //the first label will be unused and only serves to start the timer
    void tap(hipStream_t stream, std::string label){
        if constexpr (!libvship_bench) return;

        hipEvent_t newEvent;
        GPU_CHECK(hipEventCreate(&newEvent));
        GPU_CHECK(hipEventRecord(newEvent, stream));
        events.push_back(newEvent);
        labels.push_back(label);
    }
    //wait on all events
    void writeToFile(){
        if constexpr (!libvship_bench) return;

        for (const auto& event: events){
            hipEventSynchronize(event);
        }

        std::ofstream file(libvship_benchJson);
        if (!file) throw VshipError(BadPath, __FILE__, __LINE__, "Failed to open "+libvship_benchJson);

        file << '{' << std::endl;

        if (events.size() <= 1) {
            file << '}' << std::endl;
            return;
        }

        for (uint i = 1; i < events.size(); i++){
            float time;
            GPU_CHECK(hipEventElapsedTime(&time, events[i-1], events[i]));
            file << '"' << labels[i] << "\" : " << time << ',' << std::endl;
        }
        file << '}' << std::endl;
    }
};

}