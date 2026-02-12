# Parakeet Multimodal Runner: Real-Time Speech + Vision

Simultaneous real-time speech transcription and object detection using Parakeet TDT and YOLO26 with camera and microphone input.

## Overview

The multimodal runner combines two AI models running in parallel:

1. **Parakeet TDT (Audio)**: Real-time speech transcription from microphone
2. **YOLO26 (Vision)**: Object detection from camera video feed

Both streams process independently and display their outputs simultaneously:
- **Video window**: Live camera feed with bounding boxes around detected objects
- **Terminal**: Real-time transcribed text updates

## Architecture

```
┌─────────────┐                    ┌──────────────┐
│ Microphone  │───► AudioStream ───►│  Parakeet    │───► Terminal
│  (Audio)    │                    │  Transcriber │     (Text Output)
└─────────────┘                    └──────────────┘

┌─────────────┐                    ┌──────────────┐
│   Camera    │───► VideoStream ───►│    YOLO26    │───► Video Window
│  (Video)    │                    │   Detector   │     (Bounding Boxes)
└─────────────┘                    └──────────────┘
```

**Key Design Principles:**

- **Independent Streams**: Audio and video processing run concurrently in separate threads
- **Non-Blocking**: Video rendering doesn't block audio transcription
- **Modular Components**: Clean separation between capture, inference, and display
- **Real-Time Updates**: Both outputs update as new data arrives

## Prerequisites

### System Requirements

```bash
# Linux
sudo apt-get install portaudio19-dev libopencv-dev

# macOS
brew install portaudio opencv
```

### Model Files

1. **Parakeet TDT Model**: Export Parakeet ASR model (see [README.md](README.md))
2. **YOLO26 Model**: Download from [larryliu0820/models](https://huggingface.co/larryliu0820/models)

```bash
# Example: Download YOLO26n (nano) model
huggingface-cli download larryliu0820/yolo26n-ExecuTorch-XNNPACK \
  --local-dir yolo26n_model

# For quantized INT8 version (smaller, faster)
huggingface-cli download larryliu0820/yolo26n-ExecuTorch-XNNPACK-INT8 \
  --local-dir yolo26n_int8_model
```

Available YOLO26 models (by size):
- **yolo26n** (Nano): Fastest, lowest accuracy
- **yolo26s** (Small): Balanced speed/accuracy
- **yolo26m** (Medium): Good accuracy
- **yolo26l** (Large): High accuracy
- **yolo26x** (Extra Large): Best accuracy, slowest

## Building

From the executorch root directory:

```bash
# CPU/XNNPACK build
make parakeet-cpu

# Metal build (macOS)
make parakeet-metal

# CUDA build (Linux)
make parakeet-cuda
```

The build will automatically detect PortAudio and OpenCV. Both are required for the multimodal runner.

## Usage

### List Available Devices

```bash
./cmake-out/examples/models/parakeet/parakeet_multimodal_runner \
  --list_devices
```

Example output:
```
Available audio input devices:
  [0] Built-in Microphone
  [1] Logitech C920 HD Pro Webcam

Available video input devices:
  [0] Video Device 0 (1280x720)
  [1] Video Device 1 (640x480)
```

### Start Multimodal Runner

```bash
# Basic usage with default devices
./cmake-out/examples/models/parakeet/parakeet_multimodal_runner \
  --asr_model_path examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --yolo_model_path /path/to/yolo26n.pte \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model

# Specify Logitech camera for both audio and video
./cmake-out/examples/models/parakeet/parakeet_multimodal_runner \
  --asr_model_path examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --yolo_model_path /path/to/yolo26n.pte \
  --tokenizer_path examples/models/parakeet/parakeet_tdt_exports/tokenizer.model \
  --audio_device_index 1 \
  --video_device_index 1

# Use CUDA backend with custom thresholds
./cmake-out/examples/models/parakeet/parakeet_multimodal_runner \
  --asr_model_path examples/models/parakeet/parakeet_cuda/model.pte \
  --asr_data_path examples/models/parakeet/parakeet_cuda/aoti_cuda_blob.ptd \
  --yolo_model_path /path/to/yolo26s.pte \
  --yolo_data_path /path/to/yolo26s_cuda.ptd \
  --tokenizer_path examples/models/parakeet/parakeet_cuda/tokenizer.model \
  --yolo_score_threshold 0.5 \
  --yolo_nms_threshold 0.45

# Headless mode (no video window, just detection logs)
./cmake-out/examples/models/parakeet/parakeet_multimodal_runner \
  --asr_model_path model.pte \
  --yolo_model_path yolo.pte \
  --tokenizer_path tokenizer.model \
  --show_video false
```

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--asr_model_path` | Path to Parakeet .pte model | `parakeet.pte` |
| `--yolo_model_path` | Path to YOLO26 .pte model | `yolo26n.pte` |
| `--tokenizer_path` | Path to tokenizer file | `tokenizer.model` |
| `--asr_data_path` | Path to ASR .ptd data file (CUDA) | - |
| `--yolo_data_path` | Path to YOLO .ptd data file (CUDA) | - |
| `--audio_device_index` | Microphone device (-1 = default) | -1 |
| `--video_device_index` | Camera device | 0 |
| `--list_devices` | List available devices and exit | false |
| `--chunk_seconds` | Audio chunk duration | 2.0 |
| `--left_context_seconds` | Audio left context | 10.0 |
| `--yolo_score_threshold` | Detection confidence threshold | 0.45 |
| `--yolo_nms_threshold` | Non-max suppression threshold | 0.50 |
| `--show_video` | Display video window | true |

## Component APIs

### AudioStream

Handles microphone audio capture with callback-based processing.

```cpp
parakeet::AudioStreamConfig audio_config;
audio_config.sample_rate = 16000;
audio_config.channels = 1;

auto audio_stream = parakeet::create_audio_stream(audio_config);
audio_stream->set_callback([](const float* samples, size_t n) {
  // Process audio samples
});
audio_stream->open(device_index);
audio_stream->start();
```

### VideoStream

Handles camera video capture with frame-by-frame access.

```cpp
parakeet::VideoStreamConfig video_config;
video_config.width = 640;
video_config.height = 480;

auto video_stream = parakeet::create_video_stream(video_config);
video_stream->open(device_index);
video_stream->start();

cv::Mat frame;
while (video_stream->get_frame(frame)) {
  // Process frame
}
```

### YOLODetector

Performs object detection with configurable confidence thresholds.

```cpp
parakeet::YOLOConfig config;
config.class_names = parakeet::YOLODetector::get_coco_classes();
config.score_threshold = 0.45f;
config.nms_threshold = 0.50f;

parakeet::YOLODetector detector(model_path, config);
detector.initialize();

cv::Mat frame = /* camera frame */;
auto detections = detector.detect(frame);

// Draw bounding boxes
parakeet::YOLODetector::draw_detections(frame, detections);
```

### StreamingTranscriber

Manages chunked audio transcription with context buffering.

```cpp
StreamingTranscriber transcriber(
    asr_model, tokenizer, sample_rate,
    chunk_seconds, left_context_seconds);
transcriber.initialize();

// In audio callback
transcriber.process_audio_chunk(samples, num_samples);

// Get latest transcription
std::string text = transcriber.get_current_text();
```

## Performance Optimization

### Model Selection

**YOLO Model Size vs. Performance:**

| Model | Speed (FPS) | Accuracy | Use Case |
|-------|-------------|----------|----------|
| yolo26n | ~60-100 | Good | Real-time, resource-constrained |
| yolo26s | ~40-60 | Better | Balanced |
| yolo26m | ~20-40 | High | Accuracy-focused |
| yolo26l/x | ~10-20 | Highest | Offline/high-end hardware |

**Backend Selection:**

- **XNNPACK (CPU)**: Universal compatibility, moderate speed
- **Metal (macOS)**: ~2-3x faster than CPU on Apple Silicon
- **CUDA (Linux)**: ~5-10x faster than CPU on NVIDIA GPUs

### Latency Considerations

**Total Latency Components:**

1. **Audio Latency**: ~2 seconds (chunk duration) + processing time
2. **Video Latency**: ~33ms (30 FPS) + detection time
3. **Display Latency**: Minimal (hardware-dependent)

**Optimization Strategies:**

- **Reduce Audio Chunks**: Lower `--chunk_seconds` (0.5-1s) for faster ASR updates (may reduce accuracy)
- **Use Smaller YOLO Model**: yolo26n/s for higher FPS
- **Enable Quantization**: INT8 models are ~2x faster with minimal accuracy loss
- **GPU Acceleration**: Use CUDA or Metal backends

### Threading Model

The multimodal runner uses multiple threads:

1. **Main Thread**: Video capture and display loop
2. **Audio Thread**: PortAudio callback thread (microphone capture)
3. **ASR Processing Thread**: Spawned per audio chunk for transcription
4. **YOLO Processing**: Runs in main thread (synchronous with video)

## Use Cases

### Smart Meeting Assistant

Transcribe conversations while detecting objects (presentations, documents, participants).

```bash
./parakeet_multimodal_runner \
  --asr_model parakeet.pte \
  --yolo_model yolo26m.pte \
  --tokenizer tokenizer.model
```

### Robotics and Navigation

Real-time voice commands + object detection for autonomous systems.

### Accessibility

Live captions + visual scene understanding for hearing/visually impaired users.

### Security and Surveillance

Audio event detection + visual monitoring.

## Troubleshooting

**Video window not displaying:**
- Check `--show_video=true`
- Verify OpenCV installation with `pkg-config --modversion opencv4`
- Ensure X11/display server is available (Linux)

**Audio/video out of sync:**
- Normal for independent streams
- Audio has ~2s latency by design (chunked processing)
- Video is near real-time (~30 FPS)

**Low FPS:**
- Use smaller YOLO model (yolo26n)
- Enable GPU acceleration (CUDA/Metal)
- Reduce video resolution in `VideoStreamConfig`

**High memory usage:**
- Use INT8 quantized models
- Reduce `--left_context_seconds` for audio

**YOLO not detecting objects:**
- Adjust `--yolo_score_threshold` (lower = more detections)
- Ensure proper lighting
- Check camera focus

**Compilation errors:**
- Verify PortAudio: `pkg-config --modversion portaudio-2.0`
- Verify OpenCV: `pkg-config --modversion opencv4`
- Install missing dependencies

## Example: Complete Setup for Logitech C920

```bash
# 1. List devices
./parakeet_multimodal_runner --list_devices

# Output:
# Available audio input devices:
#   [0] Built-in Microphone
#   [1] Logitech C920 HD Pro Webcam
# Available video input devices:
#   [0] Built-in Camera (1280x720)
#   [1] Logitech C920 (1920x1080)

# 2. Run with Logitech C920 for both audio and video
./parakeet_multimodal_runner \
  --asr_model_path parakeet.pte \
  --yolo_model_path yolo26s.pte \
  --tokenizer_path tokenizer.model \
  --audio_device_index 1 \
  --video_device_index 1 \
  --yolo_score_threshold 0.5

# 3. Output
# === System Active ===
# Audio: 1 @ 16000Hz
# Video: 1 @ 640x480
# Press 'q' in video window to quit
#
# Transcription: hello everyone today we're going to discuss...
```

## Architecture Details

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Thread (Event Loop)                  │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ VideoStream  │─────►│ YOLODetector │                    │
│  │  (OpenCV)    │      │  (ExecuTorch)│                    │
│  └──────────────┘      └──────────────┘                    │
│         │                      │                             │
│         └──────► cv::Mat ──────┘                            │
│                    │                                         │
│                    ▼                                         │
│          ┌──────────────────┐                               │
│          │  cv::imshow()    │                               │
│          │  (Display)       │                               │
│          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Audio Thread (PortAudio Callback)               │
│                                                              │
│  ┌──────────────┐      ┌──────────────────────┐            │
│  │ AudioStream  │─────►│ StreamingTranscriber │            │
│  │ (PortAudio)  │      │   (ExecuTorch)       │            │
│  └──────────────┘      └──────────────────────┘            │
│         │                      │                             │
│         └──► Audio Buffer ─────┘                            │
│                    │                                         │
│                    ▼                                         │
│          ┌──────────────────┐                               │
│          │  std::cout       │                               │
│          │  (Terminal)      │                               │
│          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
examples/models/parakeet/
├── main.cpp                    # Original Parakeet batch runner
├── streaming_runner.cpp        # Audio-only streaming runner
├── multimodal_runner.cpp       # Combined audio + video runner
│
├── audio_stream.{h,cpp}        # Microphone capture abstraction
├── video_stream.{h,cpp}        # Camera capture abstraction
├── yolo_detector.{h,cpp}       # YOLO inference wrapper
│
├── tokenizer_utils.{h,cpp}     # Tokenizer utilities
├── timestamp_utils.{h,cpp}     # Timestamp utilities
├── types.h                     # Shared data structures
│
├── CMakeLists.txt              # Build configuration
├── README.md                   # Parakeet documentation
├── STREAMING.md                # Audio streaming guide
└── MULTIMODAL.md               # This file
```

## References

- Parakeet TDT: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- YOLO26 Models: [larryliu0820/models](https://huggingface.co/larryliu0820/models)
- PortAudio: [portaudio.com](http://www.portaudio.com/)
- OpenCV: [opencv.org](https://opencv.org/)
