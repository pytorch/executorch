#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace parakeet {

// Object detection result
struct Detection {
  int class_id = 0;
  std::string class_name;
  float confidence = 0.0f;
  cv::Rect box;
};

// YOLO detector configuration
struct YOLOConfig {
  std::vector<std::string> class_names;
  float score_threshold = 0.45f;
  float nms_threshold = 0.50f;
};

// YOLO object detector
class YOLODetector {
 public:
  YOLODetector(
      const std::string& model_path,
      const YOLOConfig& config,
      const std::string& data_path = "");

  ~YOLODetector();

  // Initialize the detector
  bool initialize();

  // Detect objects in a frame
  std::vector<Detection> detect(const cv::Mat& frame);

  // Get model input dimensions
  cv::Size get_input_size() const {
    return input_size_;
  }

  // Draw detections on a frame
  static void draw_detections(
      cv::Mat& frame,
      const std::vector<Detection>& detections,
      const cv::Scalar& color = cv::Scalar(0, 255, 0));

  // Get default COCO class names
  static std::vector<std::string> get_coco_classes();

 private:
  std::unique_ptr<::executorch::extension::Module> model_;
  YOLOConfig config_;
  cv::Size input_size_;
  bool is_initialized_;
};

} // namespace parakeet
