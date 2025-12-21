#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <numeric>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include<vector>
#include <chrono>
#include <thread>
#include<exception>
#include<set>
#include <mutex>
#include <condition_variable>
#include <cassert>

using uint = unsigned int;

#include "VshipColor.h"
#include "VshipAPI.h"

#include "ffvship_utility/ProgressBar.hpp"
#include "ffvship_utility/ffmpegmain.hpp"
#include "ffvship_utility/CLI_Parser.hpp"
#include "util/concurrency.hpp"

extern "C" {
#include <ffms.h>
#include <libavutil/pixfmt.h>
}

using score_tuple_t = std::tuple<double, double, double>;
using score_queue_t = ClosableThreadSet<std::tuple<int, score_tuple_t>>;
using frame_tuple_t = std::tuple<int, uint8_t *, uint8_t *>;
using frame_queue_t = ThreadSafeQueue<frame_tuple_t>;
using frame_pool_t = threadSet<uint8_t *>;
using ProgressBarT = ProgressBar<500>;

void frame_reader_thread(VideoManager &v1, VideoManager &v2, std::vector<int>* frames_source, std::vector<int>* frames_encoded, int threadid, int threadnum, frame_queue_t &queue,
                         frame_pool_t &frame_buffer_pool) {
    const int num_frames = frames_source->size();
    for (int i = num_frames*threadid/threadnum; i < num_frames*(threadid+1)/threadnum; i++) {
        const int source_frame = (*frames_source)[i];
        const int encoded_frame = (*frames_encoded)[i];
        uint8_t *src_buffer = frame_buffer_pool.pop();
        uint8_t *enc_buffer = frame_buffer_pool.pop();

        auto future_src =
            std::async(std::launch::async, [&v1, source_frame, src_buffer]() {
                v1.fetch_frame_into_buffer(source_frame, src_buffer);
            });

        auto future_enc =
            std::async(std::launch::async, [&v2, encoded_frame, enc_buffer]() {
                v2.fetch_frame_into_buffer(encoded_frame, enc_buffer);
            });

        future_src.get();
        future_enc.get();

        frame_tuple_t frame_tuple = std::make_tuple(i, src_buffer, enc_buffer);
        queue.push(frame_tuple);
    }
}

struct frame_reader_thread2_arguments{
    std::string source_path; std::string encoded_path;
    FFMS_Index* source_index; FFMS_Index* encoded_index;
    int source_video_track_index; int encoded_video_track_index;
    int threadid; int threadnum;
    std::vector<int>* frames_source;
    std::vector<int>* frames_encoded;
    frame_queue_t* frame_queue; frame_pool_t* frame_buffer_pool;
};

void frame_reader_thread2(frame_reader_thread2_arguments args){
    VideoManager v1(args.source_path, args.source_index, args.source_video_track_index);
    VideoManager v2(args.encoded_path, args.encoded_index, args.encoded_video_track_index);
    frame_reader_thread(v1, v2, args.frames_source, args.frames_encoded, args.threadid, args.threadnum, *args.frame_queue, *args.frame_buffer_pool);
}

void frame_worker_thread(frame_queue_t &input_queue,
                         frame_pool_t &frame_buffer_pool, GpuWorker &gpu_worker,
                         score_queue_t &output_score_queue,
                         int64_t planeSize[3],
                         int64_t planeSize2[3],
                         int* error) {
    while (!*error) {
        std::optional<std::tuple<int, uint8_t *, uint8_t *>> maybe_task =
            input_queue.pop();
        if (!maybe_task.has_value()) {
            break;
        }
        auto [frame_index, src_buffer, enc_buffer] = *maybe_task;

        const uint8_t* src_buffer_planes[3] = {src_buffer, src_buffer+planeSize[0], src_buffer+planeSize[0]+planeSize[1]};
        const uint8_t* enc_buffer_planes[3] = {enc_buffer, enc_buffer+planeSize2[0], enc_buffer+planeSize2[0]+planeSize2[1]};
        const auto& [scores, vshipError] = gpu_worker.compute_metric_score(src_buffer_planes, enc_buffer_planes);
        
        if (vshipError != Vship_NoError){
            frame_buffer_pool.insert(src_buffer);
            frame_buffer_pool.insert(enc_buffer);
            *error = 1;
            continue;
        }

        output_score_queue.insert(std::make_tuple(frame_index, scores));

        frame_buffer_pool.insert(src_buffer);
        frame_buffer_pool.insert(enc_buffer);
    }
}

void aggregate_scores_function(score_queue_t& input_score_queue,
                               std::vector<double>& aggregated_scores,
                               ProgressBarT* progressBar,
                               MetricType metric, bool livePrint) {
    while (true) {
        std::optional<std::tuple<int, std::tuple<double, double, double>>>
            maybe_score = input_score_queue.pop();

        if (!maybe_score.has_value()) {
            break;
        }

        const auto &[frame_index, scores_tuple] = *maybe_score;
        const bool should_store_first_score = (metric == MetricType::CVVDP || metric == MetricType::SSIMULACRA2);

        if (should_store_first_score) {
            aggregated_scores[frame_index] = std::get<0>(scores_tuple);
            if (progressBar) progressBar->add_value(std::get<0>(scores_tuple));
            if (livePrint) std::cout << frame_index << " " << std::get<0>(scores_tuple) << std::endl;
        } else {
            aggregated_scores[frame_index * 3] = std::get<0>(scores_tuple);
            aggregated_scores[frame_index * 3 + 1] = std::get<1>(scores_tuple);
            aggregated_scores[frame_index * 3 + 2] = std::get<2>(scores_tuple);
            if (progressBar) progressBar->add_value(std::get<2>(scores_tuple));
            if (livePrint) std::cout << frame_index << " " << std::get<0>(scores_tuple) << " " << std::get<1>(scores_tuple) << " " << std::get<2>(scores_tuple) << std::endl;
        }
    }
}

void print_aggergate_metric_statistics(const std::vector<double> &data,
                                       const std::string &label) {
    if (data.empty())
        return;

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    const size_t count = sorted.size();
    double average = 0;
    double squared_sum = 0;
    for (unsigned long i = 0; i < sorted.size(); i++){
        average += sorted[i];
    }
    average /= count;
    for (unsigned long i = 0; i < sorted.size(); i++){
        squared_sum += (sorted[i] - average)*(sorted[i] - average);
    }
    const double stddev = std::sqrt(squared_sum/count);

    std::vector<std::pair<std::string, double>> stats = {
        {"Average", average},
        {"Standard Deviation", stddev},
        {"Median", sorted[count / 2]},
        {"5th percentile", sorted[count / 20]},
        {"95th percentile", sorted[19 * count / 20]},
        {"Minimum", sorted.front()},
        {"Maximum", sorted.back()}};

    // Dynamically calculate label width
    size_t max_label_width = 0;
    for (const auto &[name, _] : stats) {
        max_label_width = std::max(max_label_width, name.size());
    }

    constexpr int value_width = 12;
    constexpr int precision = 6;
    const int spacing = 3; // " : "
    const int total_width =
        static_cast<int>(max_label_width) + spacing + value_width;

    // Center the label
    const int label_padding =
        std::max(0, (total_width - static_cast<int>(label.size())) / 2);
    std::cout << std::string(label_padding, '-') << label
              << std::string(total_width - label_padding -
                                 static_cast<int>(label.size()),
                             '-')
              << std::endl;

    for (const auto &[name, value] : stats) {
        std::cout << std::setw(static_cast<int>(max_label_width)) << std::right
                  << name << " : " << std::setw(value_width) << std::right
                  << std::fixed << std::setprecision(precision) << value
                  << std::endl;
    }

    std::cout << std::endl;
}

int main(int argc, char **argv) {
    CommandLineOptions cli_args = parse_command_line_arguments(argc, argv);
    if (cli_args.NoAssertExit){
        return 1; //error is already handled
    }

    if (cli_args.version){
        Vship_Version vshipVersion = Vship_GetVersion();
        std::cout << "FFVship 4.0.3-a" << std::endl;
        std::cout << "Repository : https://github.com/Line-fr/Vship" << std::endl;
        std::cout << "Linked against libvship version " << vshipVersion.major << "." << vshipVersion.minor << "." << vshipVersion.minorMinor << std::endl;
        if (vshipVersion.backend == Vship_Cuda){
            std::cout << "Cuda version" << std::endl;
        } else {
            std::cout << "HIP version" << std::endl;
        }
        return 0;
    }

    if (cli_args.list_gpus) {
        char errmsg[1024];
        int numgpu;
        Vship_Exception err = Vship_GetDeviceCount(&numgpu);
        if (err != Vship_NoError){
            Vship_GetDetailedLastError(errmsg, 1024);
            std::cerr << errmsg << std::endl;
            return 1;
        }
        Vship_DeviceInfo devinfo;
        for (int i = 0; i < numgpu; i++){
            std::cout << "GPU " << i << ": ";
            err = Vship_GetDeviceInfo(&devinfo, i);
            if (err != Vship_NoError){
                Vship_GetDetailedLastError(errmsg, 1024);
                std::cerr << errmsg << std::endl;
                return 1;
            }
            std::cout << devinfo.name << std::endl;
        }
        return 0;
    }

    //force -g,-t and --every at 1 when cvvdp is used because the metric is temporal
    if (cli_args.metric == MetricType::CVVDP) {
        cli_args.gpu_threads = 1;
        cli_args.cpu_threads = 1;
        cli_args.every_nth_frame = 1;
        if (!cli_args.source_indices_list.empty() || !cli_args.encoded_indices_list.empty()){
            std::cerr << "You cannot use custom indice list with CVVDP, it is a video metric and needs sequential processing" << std::endl;
            return 1;
        }
    }

    // gpu sanity check
    Vship_Exception err = Vship_GPUFullCheck(cli_args.metricParam.gpu_id);
    if (err != Vship_NoError){
        char errmsg[1024];
        Vship_GetDetailedLastError(errmsg, 1024);
        std::cerr << errmsg << std::endl;
        return 1;
    }
    /*not needed anymore we defined gpu_id in handler directly
    err = Vship_SetDevice(cli_args.gpu_id);
    if (err != Vship_NoError){
        char errmsg[1024];
        Vship_GetDetailedLastError(errmsg, 1024);
        std::cerr << errmsg << std::endl;
        return 1;
    }
    */

    auto init = std::chrono::high_resolution_clock::now();

    const uint queue_capacity = cli_args.cpu_threads;

    const uint num_gpus = cli_args.gpu_threads;
    const uint num_frame_buffer = num_gpus*2 + 2*queue_capacity + 2*cli_args.cpu_threads; //maximum number of buffers in nature possible

    FFMSIndexResult source_index = FFMSIndexResult(cli_args.source_file, cli_args.source_index, cli_args.cache_index, !cli_args.live_index_score_output, !cli_args.live_index_score_output);
    FFMSIndexResult encode_index = FFMSIndexResult(cli_args.encoded_file, cli_args.encoded_index, cli_args.cache_index, !cli_args.live_index_score_output, !cli_args.live_index_score_output);

    Vship_Colorspace_t source_colorspace;
    Vship_Colorspace_t encoded_colorspace;

    //initiliaze first sources to get width and height
    VideoManager v1(cli_args.source_file, source_index.index,
                    source_index.selected_video_track);
    int width = v1.reader->frame_width, height = v1.reader->frame_height;
    source_colorspace = v1.colorspace;

    //we want that when cropped, the encoded video matches the cropped source.
    int cropResizeWidth = width - cli_args.cropSource.left - cli_args.cropSource.right + cli_args.cropEncoded.left + cli_args.cropEncoded.right;
    int cropResizeHeight = height - cli_args.cropSource.top - cli_args.cropSource.bottom + cli_args.cropEncoded.top + cli_args.cropEncoded.bottom;

    VideoManager v2(cli_args.encoded_file, encode_index.index,
                    encode_index.selected_video_track);
    encoded_colorspace = v2.colorspace;

    source_colorspace.crop = cli_args.cropSource;
    encoded_colorspace.crop = cli_args.cropEncoded;
    encoded_colorspace.target_width = cropResizeWidth;
    encoded_colorspace.target_height = cropResizeHeight;
    //colorspaces should be complete
    if (cli_args.verbose){
        std::cout << "-------Source-Colorspace--------" << std::endl;
        printColorspace(source_colorspace);
        std::cout << "------Encoded-Colorspace--------" << std::endl;
        printColorspace(encoded_colorspace);
    }

    //sanitize start_frame, end_frame, every_nth_frame and encoded_offset
    int start = cli_args.start_frame;
    int end = cli_args.end_frame;
    int every = cli_args.every_nth_frame;
    int encoded_offset = cli_args.encoded_offset;

    if (start < 0) start = 0;
    if (end < 0) end = v1.reader->total_frame_count;
    end = std::min(end, v1.reader->total_frame_count);
    start = std::min(start, v1.reader->total_frame_count);
    if (end < start) end = start;

    //start end sanitizer for source (considering source_offset)
    start = std::max(-encoded_offset, start);
    end = start+std::min(v2.reader->total_frame_count-(start+encoded_offset), end-start);

    if (end < start){
        std::cerr << "encoded_offset " << encoded_offset << " does not allow comparing both videos" << std::endl;
        return 1;
    }

    //now we can use source_indices and encoded_indices to create a vector of all frames to cover, taking into account start end every and encoded offset
    std::vector<int> frames_source;
    std::vector<int> frames_encoded;

    if (cli_args.source_indices_list.empty()){
        frames_source.reserve((end - start + every-1)/every);
        for (int i = start; i < end; i += every) frames_source.push_back(i);
    } else {
        for (const int el : cli_args.source_indices_list){
            if (el*every + start >= v1.reader->total_frame_count){
                std::cerr << "Source Invalid frame index found " << el << " which will be placed at " << el*every+start << " after computed end " << v1.reader->total_frame_count << std::endl;
                return 1;
            }
            frames_source.push_back(start + el*every);
        }
    }

    if (cli_args.encoded_indices_list.empty() && cli_args.source_indices_list.empty()){
        frames_encoded.reserve((end - start + every-1)/every);
        for (int i = start; i < end; i += every) frames_encoded.push_back(i+encoded_offset);
    } else if (cli_args.encoded_indices_list.empty() && !cli_args.source_indices_list.empty()) {
        for (const int el : cli_args.source_indices_list){
            if (el*every + start + encoded_offset >= v2.reader->total_frame_count){
                std::cerr << "Encoded Invalid frame index found " << el << " which will be placed at " << el*every+start+encoded_offset << " after computed end " << v2.reader->total_frame_count << std::endl;
                return 1;
            }
            frames_encoded.push_back(start + el*every + encoded_offset);
        }
    } else {
        for (const int el : cli_args.encoded_indices_list){
            if (el*every + start >= end){
                std::cerr << "Encoded Invalid frame index found " << el << " which will be placed at " << el*every+start+encoded_offset << " after computed end " << end+encoded_offset << std::endl;
                return 1;
            }
            frames_encoded.push_back(start + el*every + encoded_offset);
        }
    }

    if (frames_encoded.size() != frames_source.size()){
        std::cerr << "Source indices and Encoded indices are of different sizes, aborting" << std::endl;
        return 1;
    }

    int num_frames = frames_source.size();

    if (cli_args.live_index_score_output) std::cout << num_frames << std::endl;

    std::set<uint8_t *> frame_buffers;
    for (unsigned int i = 0; i < num_frame_buffer; ++i) {
        //we allocate so that each frame may it be source or encoded can fit in
        //we use +1 for the stride because by offsetting left border, we might go to after the last line slighly. (so we allocate one more line to make it defined in memory)
        frame_buffers.insert(GpuWorker::allocate_external_rgb_buffer(std::max(v1.getBufferSize(), v2.getBufferSize())));
    }

    frame_pool_t frame_buffer_pool(frame_buffers);
    frame_queue_t frame_queue(queue_capacity);

    float source_fps = v1.reader->fps;
    if (num_frames == 1){
        source_fps = 1;
    }
    if ((cli_args.metric == MetricType::CVVDP) && (source_fps <= 0 || source_fps > 1000)){ //not acceptable range so probably an error
        std::cerr << "Warning: The source fps is either <= 0 or > 1000 which is probably an error. CVVDP needs an accurate fps value: default to 1 for now (no temporal behavior)" << std::endl;
        source_fps = 1;
    }

    std::vector<GpuWorker> gpu_workers;
    gpu_workers.reserve(num_gpus);

    for (uint i = 0; i < num_gpus; i++){
        //size of encoded version gets deduced by crops but stride needs to be given
        gpu_workers.emplace_back(cli_args.metric, source_colorspace, encoded_colorspace, v1.processor->unpack_stride, v2.processor->unpack_stride, source_fps, cli_args.metricParam);
    }

    std::vector<std::thread> reader_threads;
    reader_threads.emplace_back(frame_reader_thread, std::ref(v1), std::ref(v2), &frames_source, &frames_encoded, 0, cli_args.cpu_threads,
                            std::ref(frame_queue), std::ref(frame_buffer_pool));

    if (cli_args.cpu_threads > 1){
        frame_reader_thread2_arguments reader_args;
        reader_args.source_path = cli_args.source_file; reader_args.encoded_path = cli_args.encoded_file;
        reader_args.source_index = source_index.index; reader_args.encoded_index = encode_index.index;
        reader_args.source_video_track_index = source_index.selected_video_track; reader_args.encoded_video_track_index = encode_index.selected_video_track;
        reader_args.threadnum = cli_args.cpu_threads;
        reader_args.frames_source = &frames_source;
        reader_args.frames_encoded = &frames_encoded;
        reader_args.frame_queue = &frame_queue;
        reader_args.frame_buffer_pool = &frame_buffer_pool;

        for (int i = 1; i < cli_args.cpu_threads; i++){
            reader_args.threadid = i;
            reader_threads.emplace_back(frame_reader_thread2, reader_args);
        }
    }

    score_queue_t score_queue({});

    int error = 0;
    std::vector<std::thread> workers;
    for (uint i = 0; i < num_gpus; ++i) {
        workers.emplace_back(frame_worker_thread, std::ref(frame_queue),
                             std::ref(frame_buffer_pool),
                             std::ref(gpu_workers[i]),
                             std::ref(score_queue),
                             v1.processor->planeSizeUnpack, v2.processor->planeSizeUnpack,
                             &error);
    }

    const int score_vector_size = (cli_args.metric == MetricType::CVVDP || cli_args.metric == MetricType::SSIMULACRA2)
                                      ? num_frames
                                      : num_frames * 3;
    std::vector<double> scores(score_vector_size);
    ProgressBarT* progressBar = NULL;
    if (!cli_args.live_index_score_output) {
        std::cout << "Metric Processing Progress (Frames)" << std::endl;
        progressBar = new ProgressBarT(num_frames);
    }

    std::thread score_thread(aggregate_scores_function, std::ref(score_queue),
                             std::ref(scores), progressBar, cli_args.metric, cli_args.live_index_score_output);

    for (auto& reader_thread: reader_threads) reader_thread.join();
    frame_queue.close();

    for (auto &w : workers)
        w.join();

    score_queue.close();
    score_thread.join();

    if (error != 0){
        std::cerr << "A frame worker returned an error, this indicates some frame failed to compute, check history for an error message" << std::endl;
        return 1;
    }

    if (!cli_args.live_index_score_output){
        std::cout << std::endl; //end of progressbar
        delete progressBar;
    }

    auto fin = std::chrono::high_resolution_clock::now();

    int millitaken =
        std::chrono::duration_cast<std::chrono::milliseconds>(fin - init)
            .count();

    float fps = num_frames * 1000 / millitaken;

    // posttreatment

    // json output
    if (cli_args.json_output_file != "") {
        std::ofstream jsonfile(cli_args.json_output_file, std::ios_base::out);
        if (!jsonfile) {
            std::cerr << "Failed to open output file" << std::endl;
            return 1;
        }
        //set precision
        jsonfile << std::setprecision(15);
        jsonfile << "[";
        for (int i = 0; i < num_frames; i++) {
            jsonfile << "[";
            switch (cli_args.metric) {
            case MetricType::Butteraugli:
                jsonfile << scores[3 * i] << ", ";
                jsonfile << scores[3 * i + 1] << ", ";
                jsonfile << scores[3 * i + 2];
                break;
            case MetricType::CVVDP:
            case MetricType::SSIMULACRA2:
                jsonfile << scores[i];
                break;
            case MetricType::Unknown:
                break;
            }
            if (i == num_frames - 1) {
                jsonfile << "]";
            } else {
                jsonfile << "], ";
            }
        }
        jsonfile << "]";
    }

    if (cli_args.live_index_score_output) return 0;

    // console output
    std::cout << (cli_args.metric == MetricType::Butteraugli ? "Butteraugli"
                                                             : ((cli_args.metric == MetricType::CVVDP) ? "CVVDP" : "SSIMU2"))
              << " Result between " << cli_args.source_file << " and "
              << cli_args.encoded_file << std::endl;
    std::cout << "Computed " << num_frames << " frames at " << fps << " fps\n"
              << std::endl;

    if (cli_args.metric == MetricType::Butteraugli) {
        std::vector<double> normQ(num_frames), norm3(num_frames),
            norminf(num_frames);

        for (int i = 0; i < num_frames; ++i) {
            normQ[i] = scores[3 * i];
            norm3[i] = scores[3 * i + 1];
            norminf[i] = scores[3 * i + 2];
        }

        print_aggergate_metric_statistics(normQ, std::to_string(cli_args.metricParam.Qnorm)+"-Norm");
        print_aggergate_metric_statistics(norm3, "3-Norm");
        print_aggergate_metric_statistics(norminf, "INF-Norm");

    } else if (cli_args.metric == MetricType::SSIMULACRA2) {
        print_aggergate_metric_statistics(scores, "SSIMULACRA2");
    } else if (cli_args.metric == MetricType::CVVDP){
        const double cvvdp_val = scores[scores.size()-1];
        std::cout << "----------------CVVDP---------------" << std::endl;
        std::cout << "Video Score: " << cvvdp_val << std::endl;
    }
    return 0;
}
