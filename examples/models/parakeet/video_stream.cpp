#include "video_stream.h"

#include <iostream>

namespace parakeet {

class OpenCVVideoStream : public VideoStream {
 public:
  explicit OpenCVVideoStream(const VideoStreamConfig& config)
      : config_(config), is_active_(false) {}

  ~OpenCVVideoStream() override {
    close();
  }

  bool open(int device_index = 0) override {
    if (capture_.isOpened()) {
      std::cerr << "Video stream already open" << std::endl;
      return false;
    }

    config_.device_index = device_index;
    capture_.open(device_index);

    if (!capture_.isOpened()) {
      std::cerr << "Failed to open video device " << device_index << std::endl;
      return false;
    }

    // Set resolution
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    capture_.set(cv::CAP_PROP_FPS, config_.fps);

    // Read actual values (camera may not support requested resolution)
    config_.width = static_cast<int32_t>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.height =
        static_cast<int32_t>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    config_.fps = static_cast<int32_t>(capture_.get(cv::CAP_PROP_FPS));

    std::cout << "Video device opened: " << device_index << std::endl;
    std::cout << "Resolution: " << config_.width << "x" << config_.height
              << " @ " << config_.fps << "fps" << std::endl;

    return true;
  }

  bool start() override {
    if (!capture_.isOpened()) {
      std::cerr << "Video stream not open" << std::endl;
      return false;
    }

    is_active_ = true;
    return true;
  }

  bool stop() override {
    is_active_ = false;
    return true;
  }

  void close() override {
    if (capture_.isOpened()) {
      is_active_ = false;
      capture_.release();
    }
  }

  bool is_active() const override {
    return is_active_ && capture_.isOpened();
  }

  bool get_frame(cv::Mat& frame) override {
    if (!is_active_) {
      return false;
    }

    return capture_.read(frame);
  }

  const VideoStreamConfig& get_config() const override {
    return config_;
  }

 private:
  VideoStreamConfig config_;
  cv::VideoCapture capture_;
  bool is_active_;
};

std::unique_ptr<VideoStream> create_video_stream(
    const VideoStreamConfig& config) {
  return std::make_unique<OpenCVVideoStream>(config);
}

std::vector<std::string> VideoStream::list_devices() {
  std::vector<std::string> devices;

  // Try to open up to 10 devices
  for (int i = 0; i < 10; i++) {
    cv::VideoCapture test_cap(i);
    if (test_cap.isOpened()) {
      int width = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_WIDTH));
      int height = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      devices.push_back(
          std::string("[") + std::to_string(i) + "] Video Device " +
          std::to_string(i) + " (" + std::to_string(width) + "x" +
          std::to_string(height) + ")");
      test_cap.release();
    }
  }

  return devices;
}

} // namespace parakeet
