#pragma once

#include <chrono>
#include <iostream>
#include <sstream>

//https://stackoverflow.com/questions/23369503/get-size-of-terminal-window-rows-columns

#if defined(_WIN32)
#elif defined(__linux__)
#include <sys/ioctl.h>
#endif // Windows/Linux

void get_terminal_size(int& width, int& height) {
#if defined(_WIN32)
    width = 50;
    height = 30;
#elif defined(__linux__)
    struct winsize w;
    ioctl(fileno(stdout), TIOCGWINSZ, &w);
    width = (int)(w.ws_col);
    height = (int)(w.ws_row);
#endif // Windows/Linux
}

//not threadsafe now
//refreshrate in ms, it represent minimal time but we wait for a frame to refresh
template<int RefreshRate = 500, bool DisplayValue = true>
class ProgressBar{
    double total_sum = 0;
    int64_t num = 0;
    int64_t total = 0;
    std::chrono::time_point<std::chrono::steady_clock> timeinit = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> firsttime = std::chrono::steady_clock::now();
public:
    ProgressBar(int64_t total) : total(total){
        refresh(true);
    }
    void refresh(bool force = false){
        const auto timeend = std::chrono::steady_clock::now();
        const auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(timeend - timeinit);
        if (!force && milli.count() < RefreshRate && (!(num == total))) return;
        timeinit = timeend;
        const auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeend - firsttime);

        int termWidth, termHeight;
        get_terminal_size(termWidth, termHeight);
        termWidth = std::min(termWidth, 100);

        std::stringstream ss;

        const double fps = (double)num*1000/total_elapsed.count();
        ss << "] " << num << "/" << total;
        if constexpr (DisplayValue){
            ss << " Avg : " << std::fixed << std::setprecision(2) << (double)total_sum/num;
        }
        ss << " IPS: " << fps;


        const int barwidth = termWidth-ss.str().size()-1;

        std::cout << '\r' << "[";
        for (int i = 0; i < barwidth; i++){
            if (i*total <= num*barwidth){
                std::cout << "|";
            } else {
                std::cout << " ";
            }
        }
        std::cout << ss.str() << std::flush;
    }
    void add_value(double val){
        total_sum += val;
        num++;
        refresh();
    }
    void set_values(int64_t num, int64_t total){
        this->num = num;
        this->total = total;
        refresh();
    }
};