"""Configuration for the public SafeSABR Starlink-ABR code path.

The public repository keeps only the Starlink high-bitrate setting used by
SafeSABR.  StarNet measurement data are not redistributed; see docs/DATA.md
for the expected trace-preparation workflow.
"""

DATASET_NAME = "starlink_high"

VIDEO_BIT_RATE = [3000, 8000, 15000, 30000, 60000, 120000]  # Kbps
REBUF_PENALTY = 40

TRAIN_TRACES = [
    "./video_trace/trace/starlink/us/train",
    "./video_trace/trace/starlink/osn/train",
    "./video_trace/trace/starlink/vic/train",
]

FINE_TUNE_TRACES = [
    "./video_trace/trace/starlink/us/calib",
    "./video_trace/trace/starlink/osn/calib",
    "./video_trace/trace/starlink/vic/calib",
]

TEST_TRACES = [
    "./video_trace/trace/starlink/us/test",
    "./video_trace/trace/starlink/osn/test",
    "./video_trace/trace/starlink/vic/test",
]

VIDEO_SIZE_FILE = "./video_trace/video/starlink_4k8k_synth/video_size_"
LOG_FILE_DIR = "./test_results/starlink_high/"

