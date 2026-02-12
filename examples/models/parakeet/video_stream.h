#pragma once

#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace parakeet {

// Video configuration for streaming
struct VideoStreamConfig {
  int32_t width = 640;
  int32_t height = 480;
  int32_t fps = 30;
  int32_t device_index = 0;
};

// Callback invoked when a new frame is available
// Parameters: cv::Mat frame
using FrameCallback = std::function<void(const cv::Mat&)>;

// Interface for video stream from camera
class VideoStream {
 public:
  virtual ~VideoStream() = default;

  // Open and start capturing video from camera
  // device_index: 0 for default camera, or specific device ID
  virtual bool open(int device_index = 0) = 0;

  // Start streaming video
  virtual bool start() = 0;

  // Stop streaming video
  virtual bool stop() = 0;

  // Close the video stream
  virtual void close() = 0;

  // Check if stream is active
  virtual bool is_active() const = 0;

  // Get next frame (blocking)
  virtual bool get_frame(cv::Mat& frame) = 0;

  // Get current configuration
  virtual const VideoStreamConfig& get_config() const = 0;

  // List available video input devices
  static std::vector<std::string> list_devices();
};

// Factory function to create a video stream
std::unique_ptr<VideoStream> create_video_stream(
    const VideoStreamConfig& config);

} // namespace parakeet
