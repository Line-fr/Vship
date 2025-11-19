#pragma once

extern "C" {
#include <ffms.h>
#include <libavutil/pixfmt.h>
}

// #include "util/CLI_Parser.hpp"
#include "ffmpegToVshipColorFormat.hpp"
#include "unpack.hpp"
#include "../util/preprocessor.hpp"

#include <cstdio>
#include <cstdlib>
#include <optional>

#ifndef ASSERT_WITH_MESSAGE
#define ASSERT_WITH_MESSAGE(condition, message)\
if (!(condition)) {\
    std::fprintf(stderr, "Assertion failed!\nExpression : %s\nFile       : %s\n  Line       : %d\nMessage    : %s\n", #condition, __FILE__, __LINE__, message);\
    std::abort();\
}
#endif

enum class MetricType { SSIMULACRA2, Butteraugli, CVVDP, Unknown };

struct MetricParameters{
    //CVVDP
    bool resizeToDisplay = 0;
    std::string model_key = "standard_fhd";

    //butteraugli
    int intensity_target_nits = 203;
    int Qnorm = 2;
};

class GpuWorker {
  private:
    Vship_Colorspace_t image_colorspace;
    int64_t lineSize[3];

    Vship_Colorspace_t encoded_colorspace;
    int64_t lineSize2[3];

    MetricType selected_metric;

    ssimu2::SSIMU2ComputingImplementation ssimu2worker;
    butter::ButterComputingImplementation butterworker;
    cvvdp::CVVDPComputingImplementation cvvdpworker;

  public:
    GpuWorker(MetricType metric, Vship_Colorspace_t source_colorspace, Vship_Colorspace_t encoded_colorspace, const int64_t lineSize[3], const int64_t lineSize2[3], float fps, MetricParameters metricParam){
        selected_metric = metric;
        this->image_colorspace = source_colorspace;
        this->encoded_colorspace = encoded_colorspace;
        for (int i = 0; i < 3; i++){
            this->lineSize[i] = lineSize[i];
            this->lineSize2[i] = lineSize2[i];
        }
        allocate_gpu_memory(fps, metricParam);
    }
    ~GpuWorker(){
        deallocate_gpu_memory();
    }

    std::tuple<float, float, float>
    compute_metric_score(const uint8_t* srcp1[3], const uint8_t* srcp2[3]) {

        if (selected_metric == MetricType::SSIMULACRA2) {
            const float s = ssimu2worker.run(srcp1, srcp2, lineSize, lineSize2); 
            return {s, s, s};
        }

        if (selected_metric == MetricType::Butteraugli) {
            return butterworker.run(nullptr, 0, srcp1, srcp2, lineSize, lineSize2);
        }

        if (selected_metric == MetricType::CVVDP){
            const float s = cvvdpworker.run(nullptr, 0, srcp1, srcp2, lineSize, lineSize2);
            return {s, s, s};
        }

        ASSERT_WITH_MESSAGE(false, "Unknown metric specified for GpuWorker.");
        return {0.0f, 0.0f, 0.0f};
    }

    static uint8_t *allocate_external_rgb_buffer(int bytes) {
        const size_t buffer_size_bytes = bytes * 3;
        uint8_t *buffer_ptr = nullptr;

        const hipError_t result = hipHostMalloc(
            reinterpret_cast<void **>(&buffer_ptr), buffer_size_bytes);

        ASSERT_WITH_MESSAGE(
            result == hipSuccess && buffer_ptr != nullptr,
            "Pinned buffer allocation failed in allocate_external_rgb_buffer");

        return buffer_ptr;
    }

    static void deallocate_external_buffer(uint8_t *buffer_ptr) {
        if (buffer_ptr != nullptr) {
            hipHostFree(buffer_ptr);
        }
    }

  private:
    void allocate_gpu_memory(float fps, MetricParameters metricParam) {
        if (selected_metric == MetricType::SSIMULACRA2) {
            try {
                ssimu2worker.init(image_colorspace, encoded_colorspace);
            } catch (const VshipError& e){
                std::cerr << e.getErrorMessage() << std::endl;
                ASSERT_WITH_MESSAGE(false, "Failed to initialize SSIMULACRA2 Worker");
                return;
            }
        } else if (selected_metric == MetricType::Butteraugli) {
            try {
                butterworker.init(image_colorspace, encoded_colorspace, metricParam.Qnorm, metricParam.intensity_target_nits);
            } catch (const VshipError& e){
                std::cerr << e.getErrorMessage() << std::endl;
                ASSERT_WITH_MESSAGE(false, "Failed to initialize Butteraugli Worker");
                return;
            }
        } else if (selected_metric == MetricType::CVVDP){
            try {
                cvvdpworker.init(image_colorspace, encoded_colorspace, fps, metricParam.resizeToDisplay, metricParam.model_key);
            } catch (const VshipError& e){
                std::cerr << e.getErrorMessage() << std::endl;
                ASSERT_WITH_MESSAGE(false, "Failed to initialize CVVDP Worker");
                return;
            }
        } else {
            ASSERT_WITH_MESSAGE(false,
                                "Unknown metric during memory allocation.");
        }
    }

    void deallocate_gpu_memory() {
        if (selected_metric == MetricType::SSIMULACRA2) {
            ssimu2worker.destroy();
        } else if (selected_metric == MetricType::Butteraugli) {
            butterworker.destroy();
        } else if (selected_metric == MetricType::CVVDP){
            cvvdpworker.destroy();
        }
    }
};

int IndexCallback(int64_t current, int64_t total, void* progp){
    auto prog = (ProgressBar<100, false>*)progp;
    prog->set_values(current/1000000, total/1000000);
    return 0;
}

class FFMSIndexResult {
  private:
    static constexpr int error_message_buffer_size = 1024;

    char error_message_buffer[error_message_buffer_size]{};
    FFMS_ErrorInfo error_info;

  public:
    std::string file_path;
    std::string index_file_path;
    FFMS_Index* index = NULL;
    FFMS_Track* track = NULL;
    int selected_video_track = -1;
    int numFrame = 0;
    int write = 0;

    explicit FFMSIndexResult(const std::string& input_file_path, std::string input_index_file_path, const bool cache_index, const bool debug_out = false, const bool show_progress = true) {
        file_path = input_file_path;
        index_file_path = input_index_file_path;
        FFMS_Init(0, 0);

        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = error_message_buffer_size;
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;

        bool from_file_success = false;

        //if cached but no path specified, we default to this path
        if (input_index_file_path == "" && cache_index) input_index_file_path = input_file_path + ".ffindex";

        //if path not empty, we try to read
        if (input_index_file_path != ""){
            index = FFMS_ReadIndex(input_index_file_path.c_str(), &error_info);
            if (index != nullptr && !FFMS_IndexBelongsToFile(index, input_file_path.c_str(), &error_info)) {
                from_file_success = true;
                if (debug_out) std::cout << "Successfully read index from [" << input_index_file_path << "]" << std::endl;
            } else {
                if (debug_out) std::cout << "Index file at [" << input_index_file_path << "] is invalid or does not exist, creating" << std::endl;
            }
        }

        //if failed, we will need to compute ourself
        if (!from_file_success){
            FFMS_Indexer *indexer = FFMS_CreateIndexer(input_file_path.c_str(), &error_info);
            ASSERT_WITH_MESSAGE(indexer != nullptr,
                            ("FFMS2: Failed to create indexer for file [" +
                            input_file_path + "] - " + error_message_buffer)
                                .c_str());

            ProgressBar<100, false>* prog;
            if (show_progress){
                std::cout << "Indexing Progress (MB) " << input_file_path << std::endl;
                prog = new ProgressBar<100, false>(0);
                FFMS_SetProgressCallback(indexer, IndexCallback, prog);
            }

            index = FFMS_DoIndexing2(indexer, FFMS_IEH_ABORT, &error_info);
            ASSERT_WITH_MESSAGE(index != nullptr,
                            ("FFMS2: Failed to index file [" + input_file_path +
                            "] - " + error_message_buffer)
                                .c_str());
            
            if (show_progress){
                prog->refresh(true);
                delete prog;
                std::cout << std::endl; //end of progressbar
            }
        }

        //if we need to cache and it s computed (compiler will optimize automatically)
        //we will write the cache file
        if (!from_file_success && cache_index){
            write = FFMS_WriteIndex(input_index_file_path.c_str(), index, &error_info);
            ASSERT_WITH_MESSAGE(write == 0,
                                ("FFMS2: Failed to write index to file [" + input_index_file_path +
                                "] - " + error_message_buffer)
                                    .c_str());
            if (debug_out) std::cout << "Successfully wrote index to [" << input_index_file_path << "]" << std::endl;
        }

        selected_video_track =
            FFMS_GetFirstTrackOfType(index, FFMS_TYPE_VIDEO, &error_info);
        ASSERT_WITH_MESSAGE(selected_video_track >= 0,
                            ("FFMS2: No video track found in file [" +
                             input_file_path + "] - " + error_message_buffer)
                                .c_str());

        track =
            FFMS_GetTrackFromIndex(index, selected_video_track);
        ASSERT_WITH_MESSAGE(track != NULL,
                            ("FFMS2: Failed to get FFMS_Track in file [" +
                             input_file_path + "]").c_str());

        numFrame = FFMS_GetNumFrames(track);
        ASSERT_WITH_MESSAGE(numFrame != 0, ("FFMS2: Got 0 frames in [" +
                             input_file_path + "]").c_str());

    }

    std::set<int> getKeyFrameIndices(){
        std::set<int> out;
        for (int i = 0; i < numFrame; i++){
            const FFMS_FrameInfo* frameinfo = FFMS_GetFrameInfo(track, i);
            ASSERT_WITH_MESSAGE(frameinfo != NULL, ("Failed to retrieve KeyFrame information in the indexer for file [" + file_path + "]").c_str());
            if (frameinfo->KeyFrame != 0){
                out.insert(i);
            }
        }
        return std::move(out);
    }

    ~FFMSIndexResult() {
        if (index != nullptr) {
            FFMS_DestroyIndex(index);
            index = nullptr;
        }

        selected_video_track = -1;
    }
};

class FFMSFrameReader {
  private:
    int num_decoder_threads = std::thread::hardware_concurrency();
    int seek_mode = FFMS_SEEK_NORMAL;

  public:
    const FFMS_VideoProperties *video_properties = nullptr;
    const FFMS_Frame *current_frame = nullptr;

    AVPixelFormat video_pixel_format = AV_PIX_FMT_NONE;
    int frame_width = 0;
    int frame_height = 0;
    int total_frame_count = 0;
    float fps = 0;

    FFMS_VideoSource *video_source = nullptr;

    FFMS_ErrorInfo error_info;
    char error_message_buffer[1024] = {};

    explicit FFMSFrameReader(const std::string &file_path, FFMS_Index *index,
                             int video_track_index) {
        initialize_error_info();
        create_video_source(file_path, index, video_track_index);
        load_video_properties();
        load_first_frame();
        configure_pixel_format(file_path);
    }

    ~FFMSFrameReader() {
        if (video_source != nullptr) {
            FFMS_DestroyVideoSource(video_source);
            video_source = nullptr;
        }
    }

    void fetch_frame(int frame_index) {
        current_frame = FFMS_GetFrame(video_source, frame_index, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch frame [" +
                             std::to_string(frame_index) + "] - " +
                             error_message_buffer)
                                .c_str());
    }

  private:
    void initialize_error_info() {
        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = sizeof(error_message_buffer);
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;
    }

    void create_video_source(const std::string &file_path, FFMS_Index *index,
                             int track_index) {

        video_source =
            FFMS_CreateVideoSource(file_path.c_str(), track_index, index,
                                   num_decoder_threads, seek_mode, &error_info);

        ASSERT_WITH_MESSAGE(video_source != nullptr,
                            ("FFMS2: Failed to create video source for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }

    void load_video_properties() {
        video_properties = FFMS_GetVideoProperties(video_source);
        ASSERT_WITH_MESSAGE(video_properties != nullptr,
                            "FFMS2: FFMS_GetVideoProperties returned null.");

        total_frame_count = video_properties->NumFrames;
        ASSERT_WITH_MESSAGE(total_frame_count > 0,
                            "FFMS2: No frames found in video stream.");
        
        fps = (float)video_properties->FPSNumerator / (float)video_properties->FPSDenominator;
    }

    void load_first_frame() {
        current_frame = FFMS_GetFrame(video_source, 0, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch first frame - " +
                             std::string(error_message_buffer))
                                .c_str());

        video_pixel_format =
            static_cast<AVPixelFormat>(current_frame->EncodedPixelFormat);
        frame_width = current_frame->EncodedWidth;
        frame_height = current_frame->EncodedHeight;
    }

    void configure_pixel_format(const std::string &file_path) {
        int output_formats[] = {static_cast<int>(video_pixel_format), -1};

        int result = FFMS_SetOutputFormatV2(video_source, output_formats,
                                            frame_width, frame_height,
                                            FFMS_RESIZER_BICUBIC, &error_info);

        ASSERT_WITH_MESSAGE(result == 0,
                            ("FFMS2: Failed to set output format for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }
};

class BasicConverterProcessor {
  public:
    AVPixelFormat src_pixfmt;
    int64_t width;
    int64_t height;
    int64_t unpack_stride[3];
    int64_t planeSizeUnpack[3];
    bool require_unpack = true;

    BasicConverterProcessor(const FFMS_Frame *ref_frame) {
        initialize_formats(ref_frame);
        build_unpack(ref_frame);
    }

    ~BasicConverterProcessor() {
    }

    //dst planes should have size given by planeSizeUnpack
    void process(const FFMS_Frame *src, uint8_t *dst[3]) {
        unpack(src, dst);
    }

  private:
    void initialize_formats(const FFMS_Frame *frame) {

        width = frame->EncodedWidth;
        height = frame->EncodedHeight;
        src_pixfmt = (AVPixelFormat)frame->EncodedPixelFormat;
    }

    void build_unpack(const FFMS_Frame *frame){
        //list supported formats (they are repeated in ffmpegToZimgFormat to handle them correctly)
        require_unpack = true;
        switch (src_pixfmt){

            //bitdepth 8, 422
            case AV_PIX_FMT_YUYV422:
            unpack_stride[0] = width;
            unpack_stride[1] = (width/2);
            unpack_stride[2] = unpack_stride[1]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            //bitdepth 8, 444
            case AV_PIX_FMT_RGB24:
            case AV_PIX_FMT_ARGB:
            case AV_PIX_FMT_RGBA:
            case AV_PIX_FMT_ABGR:
            case AV_PIX_FMT_BGRA:
            unpack_stride[0] = width;
            unpack_stride[1] = unpack_stride[0];
            unpack_stride[2] = unpack_stride[0]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            //depth 16, 444
            case AV_PIX_FMT_RGB48LE:
            case AV_PIX_FMT_RGBA64LE:
            unpack_stride[0] = width*2;
            unpack_stride[1] = unpack_stride[0];
            unpack_stride[2] = unpack_stride[0]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            default:
                Vship_Colorspace_t colorspace;
                ffmpegToVshipFormat(colorspace, frame);
                unpack_stride[0] = frame->Linesize[0];
                unpack_stride[1] = frame->Linesize[1];
                unpack_stride[2] = frame->Linesize[2];
                planeSizeUnpack[0] = unpack_stride[0]*height;
                planeSizeUnpack[1] = (height >> colorspace.subsampling.subh)*unpack_stride[1];
                planeSizeUnpack[2] = (height >> colorspace.subsampling.subh)*unpack_stride[2];
                require_unpack = false;
                return; //no unpack
        }
    }

    void unpack(const FFMS_Frame *src, uint8_t* unpack_buffer[3]){
        if (!require_unpack){
            memcpy(unpack_buffer[0], src->Data[0], planeSizeUnpack[0]);
            memcpy(unpack_buffer[1], src->Data[1], planeSizeUnpack[1]);
            memcpy(unpack_buffer[2], src->Data[2], planeSizeUnpack[2]);
            return;
        }
        switch (src_pixfmt){
            case AV_PIX_FMT_YUYV422:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width/2; i++){
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        //U
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i+1] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        //V
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_RGB48LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA64LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+2];
                    }
                }
            break;

            case AV_PIX_FMT_RGB24:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_ARGB:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_ABGR:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                    }
                }
            break;
            case AV_PIX_FMT_BGRA:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+0];
                    }
                }
            break;

            default:
                //this should not happen and result of forgiveness of the dev, better to place this in case
                std::cout << "Error: Trying to unpack an unsupported format " << src_pixfmt << " line " << __LINE__ << " of " << __FILE__ << std::endl;
                return;
        }
    }
};

class VideoManager {
  public:
    Vship_Colorspace_t colorspace;
    std::unique_ptr<FFMSFrameReader> reader;
    std::unique_ptr<BasicConverterProcessor> processor;

    VideoManager(const std::string &file_path, FFMS_Index *index, int video_track_index) {

        reader = std::make_unique<FFMSFrameReader>(file_path, index,
                                                   video_track_index);

        processor = std::make_unique<BasicConverterProcessor>(reader->current_frame);

        ffmpegToVshipFormat(colorspace, reader->current_frame);

        ASSERT_WITH_MESSAGE(
            processor != nullptr,
            "VideoManager: Failed to initialize ZimgProcessor.");
    }
    int64_t getBufferSize(){
        return processor->planeSizeUnpack[0]+processor->planeSizeUnpack[1]+processor->planeSizeUnpack[2];
    }
    void fetch_frame_into_buffer(int frame_index, uint8_t *output_buffer) {
        reader->fetch_frame(frame_index);
        uint8_t* srcp[3] = {output_buffer, output_buffer+processor->planeSizeUnpack[0], output_buffer+processor->planeSizeUnpack[0]+processor->planeSizeUnpack[1]};
        processor->process(reader->current_frame, srcp);
    }
};

void printColorspace(Vship_Colorspace_t colorspace){
    std::cout << "Source Size: " << colorspace.width << "x" << colorspace.height << std::endl;
    std::cout << "Resize To: " << colorspace.target_width << "x" << colorspace.target_height << std::endl;
    std::cout << "Cropped by (Top/Bottom/Left/Right): " << colorspace.crop.top << "/" << colorspace.crop.bottom << "/" << colorspace.crop.left << "/" << colorspace.crop.right << std::endl;
    std::cout << "Converted Size: ";
    if (colorspace.target_width != -1){ 
        std::cout << colorspace.target_width-colorspace.crop.right-colorspace.crop.left;
    } else {
        std::cout << colorspace.width-colorspace.crop.right-colorspace.crop.left;
    }
    std::cout << "x";
    if (colorspace.target_height != -1){
        std::cout << colorspace.target_height-colorspace.crop.top-colorspace.crop.bottom;
    } else {
        std::cout << colorspace.height-colorspace.crop.top-colorspace.crop.bottom;
    }
    std::cout << std::endl;
    std::cout << "Sample Type: ";
    switch(colorspace.sample){
        case Vship_SampleUINT8:
            std::cout << "Uint8_t";
            break;
        case Vship_SampleUINT9:
            std::cout << "Uint9_t";
            break;
        case Vship_SampleUINT10:
            std::cout << "Uint10_t";
            break;
        case Vship_SampleUINT12:
            std::cout << "Uint12_t";
            break;
        case Vship_SampleUINT14:
            std::cout << "Uint14_t";
            break;
        case Vship_SampleUINT16:
            std::cout << "Uint16_t";
            break;
        case Vship_SampleHALF:
            std::cout << "Half";
            break;
        case Vship_SampleFLOAT:
            std::cout << "Float";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Color Family: ";
    if (colorspace.colorFamily == Vship_ColorRGB){
        std::cout << "RGB";
    } else if (colorspace.colorFamily == Vship_ColorYUV){
        std::cout << "YUV";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Range: ";
    if (colorspace.range == Vship_RangeLimited){
        std::cout << "Limited";
    } else if (colorspace.range == Vship_RangeFull){
        std::cout << "Full";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Subsampling (log): " << colorspace.subsampling.subw << "x" << colorspace.subsampling.subh << std::endl;
    std::cout << "Chroma Location: ";
    switch(colorspace.chromaLocation){
        case Vship_ChromaLoc_Center:
            std::cout << "Center";
            break;
        case Vship_ChromaLoc_TopLeft:
            std::cout << "Top-Left";
            break;
        case Vship_ChromaLoc_Left:
            std::cout << "Left";
            break;
        case Vship_ChromaLoc_Top:
            std::cout << "Top";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "YUV Matrix: ";
    switch(colorspace.YUVMatrix){
        case Vship_MATRIX_RGB:
            std::cout << "RGB";
            break;
        case Vship_MATRIX_BT709:
            std::cout << "BT709";
            break;
        case Vship_MATRIX_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_MATRIX_ST170_M:
            std::cout << "ST170_M";
            break;
        case Vship_MATRIX_BT2020_NCL:
            std::cout << "BT2020_NCL";
            break;
        case Vship_MATRIX_BT2020_CL:
            std::cout << "BT2020_CL";
            break;
        case Vship_MATRIX_BT2100_ICTCP:
            std::cout << "BT2100_ICTCP";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Transfer Function: ";
    switch(colorspace.transferFunction){
        case Vship_TRC_BT709:
            std::cout << "BT709";
            break;
        case Vship_TRC_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_TRC_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_TRC_BT601:
            std::cout << "BT601";
            break;
        case Vship_TRC_Linear:
            std::cout << "Linear";
            break;
        case Vship_TRC_sRGB:
            std::cout << "sRGB";
            break;
        case Vship_TRC_PQ:
            std::cout << "PQ";
            break;
        case Vship_TRC_ST428:
            std::cout << "ST428";
            break;
        case Vship_TRC_HLG:
            std::cout << "HLG";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Primaries: ";
    switch(colorspace.primaries){
        case Vship_PRIMARIES_INTERNAL:
            std::cout << "XYZ";
            break;
        case Vship_PRIMARIES_BT709:
            std::cout << "BT709";
            break;
        case Vship_PRIMARIES_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_PRIMARIES_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_PRIMARIES_BT2020:
            std::cout << "BT2020";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
}

struct CommandLineOptions {
    std::string source_file;
    std::string encoded_file;
    std::string json_output_file;
    std::string source_index;
    std::string encoded_index;

    int start_frame = 0;
    int end_frame = -1;
    int every_nth_frame = 1;
    int encoded_offset = 0;

    Vship_CropRectangle_t cropSource;
    Vship_CropRectangle_t cropEncoded;

    std::vector<int> source_indices_list;
    std::vector<int> encoded_indices_list;

    MetricParameters metricParam;

    int gpu_id = 0;
    int gpu_threads = 3;
    int cpu_threads = 1;

    bool list_gpus = false;
    bool version = false;
    bool verbose = false;
    MetricType metric = MetricType::SSIMULACRA2; //SSIMULACRA2 by default

    bool NoAssertExit = false; //please exit without creating an assertion failed scary error

    bool live_index_score_output = false;

    bool cache_index = false;
};

std::vector<int> splitPerToken(std::string inp){
    std::vector<int> out;
    std::string temp;

    for (const char c: inp){
        switch (c){
            case ',':
                out.push_back(std::stoi(temp));
                temp.clear();
                break;
            case ' ':
                continue;
            default:
                temp.push_back(c);
        }
    }
    if (!temp.empty()) out.push_back(std::stoi(temp));

    return out;
}

MetricType parse_metric_name(const std::string &name) {
    std::string lowered;
    lowered.resize(name.size());
    for (unsigned int i = 0; i < name.size(); i++){
        lowered[i] = std::tolower(name[i]);
    }
    if (lowered == "ssimulacra2" || lowered == "ssimu2") return MetricType::SSIMULACRA2;
    if (lowered == "butteraugli" || lowered == "butter") return MetricType::Butteraugli;
    if (lowered == "cvvdp") return MetricType::CVVDP;
    return MetricType::Unknown;
}

CommandLineOptions parse_command_line_arguments(int argc, char **argv) {
    std::vector<std::string> args(argc);
    for (int i = 0; i < argc; i++){
        args[i] = argv[i];
    }

    helper::ArgParser parser;

    std::string metric_name;
    std::string source_indices_str;
    std::string encoded_indices_str;

    CommandLineOptions opts;

    parser.add_flag({"--source", "-s"}, &opts.source_file, "Reference video to compare to", true);
    parser.add_flag({"--encoded", "-e"}, &opts.encoded_file, "Distorted encode of the source", true);
    parser.add_flag({"--metric", "-m"}, &metric_name, "Which metric to use [SSIMULACRA2, Butteraugli, CVVDP]");
    parser.add_flag({"--json"}, &opts.json_output_file, "Outputs metric results to a json file");
    parser.add_flag({"--live-score-output"}, &opts.live_index_score_output, "replace stdout output with index-score lines");
    parser.add_flag({"--source-index"}, &opts.source_index, "FFMS2 index file for source video");
    parser.add_flag({"--encoded-index"}, &opts.encoded_index, "FFMS2 index file for encoded video");
    parser.add_flag({"--cache-index"}, &opts.cache_index, "Write index files to disk and reuse if available");

    parser.add_flag({"--start"}, &opts.start_frame, "Starting frame of source");
    parser.add_flag({"--end"}, &opts.end_frame, "Ending frame of source");
    parser.add_flag({"--encoded-offset"}, &opts.encoded_offset, "Frame offset of encoded video to source");
    parser.add_flag({"--every"}, &opts.every_nth_frame, "Frame sampling rate");
    parser.add_flag({"--source-indices"}, &source_indices_str, "List of source indices subjective to --start, --end, --every and --encoded-offset. If --encoded-indices isnt specified, this will be applied to encoded-indices too. Format is integers separated by comma");
    parser.add_flag({"--encoded-indices"}, &encoded_indices_str, "List of encoded indices subjective to --start, --end, --every and --encoded-offset. Format is integers separated by comma");
    
    parser.add_flag({"--cropTopSource"}, &opts.cropSource.top, "Allows to crop source");
    parser.add_flag({"--cropBottomSource"}, &opts.cropSource.bottom, "Allows to crop source");
    parser.add_flag({"--cropLeftSource"}, &opts.cropSource.left, "Allows to crop source");
    parser.add_flag({"--cropRightSource"}, &opts.cropSource.right, "Allows to crop source");
    parser.add_flag({"--cropTopEncoded"}, &opts.cropEncoded.top, "Allows to crop Encoded");
    parser.add_flag({"--cropBottomEncoded"}, &opts.cropEncoded.bottom, "Allows to crop Encoded");
    parser.add_flag({"--cropLeftEncoded"}, &opts.cropEncoded.left, "Allows to crop Encoded");
    parser.add_flag({"--cropRightEncoded"}, &opts.cropEncoded.right, "Allows to crop Encoded");

    parser.add_flag({"--resizeToDisplay"}, &opts.metricParam.resizeToDisplay, "Allow to resize to the screen resolution specified in the model (default off)");
    parser.add_flag({"--displayModel"}, &opts.metricParam.model_key, "Allow specifying screen disposition to CVVDP (default standard_fhd)");
    parser.add_flag({"--intensity-target"}, &opts.metricParam.intensity_target_nits, "Target nits for Butteraugli");
    parser.add_flag({"--qnorm"}, &opts.metricParam.Qnorm, "Optional Norm to compute (default to 2)");
    parser.add_flag({"--threads", "-t"}, &opts.cpu_threads, "Number of Decoder process, recommended is 2");
    parser.add_flag({"--gpu-threads", "-g"}, &opts.gpu_threads, "GPU thread count, recommended is 3");
    parser.add_flag({"--gpu-id"}, &opts.gpu_id, "GPU index");
    parser.add_flag({"--list-gpu"}, &opts.list_gpus, "List available GPUs");
    parser.add_flag({"--version"}, &opts.version, "Print FFVship version");
    parser.add_flag({"--verbose"}, &opts.verbose, "Print Colorspace found in Source and encoded");

    if (parser.parse_cli_args(args) != 0) { //the parser will have already printed an error
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.list_gpus || opts.version) return opts;

    try {
        opts.source_indices_list = splitPerToken(source_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --source-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }
    try {
        opts.encoded_indices_list = splitPerToken(encoded_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --encoded-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.source_file.empty()){
        std::cerr << "Source file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (opts.encoded_file.empty()){
        std::cerr << "Encoded file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (!metric_name.empty()) {
        opts.metric = parse_metric_name(metric_name);
        if (opts.metric == MetricType::Unknown){
            std::cerr << "Unknown metric type. Expected 'SSIMULACRA2' or 'Butteraugli'." << std::endl;
            opts.NoAssertExit = true;
        }
    }

    return opts;
}
