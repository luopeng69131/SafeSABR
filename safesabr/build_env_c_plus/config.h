#ifndef SAFESABR_CONFIG_H
#define SAFESABR_CONFIG_H

#include <string>

const int BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {
    3000, 8000, 15000, 30000, 60000, 120000
};
const double REBUF_PENALTY = 40.0;

const std::string TEST_TRACES = "./video_trace/trace/starlink/";
const std::string VIDEO_SIZE_FILE =
    "./video_trace/video/starlink_4k8k_synth/video_size_";
const std::string LOG_FILE_DIR = "./test_results/starlink_high/";

#endif

