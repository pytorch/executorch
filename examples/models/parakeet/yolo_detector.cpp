#include "yolo_detector.h"

#include <iostream>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

using ::executorch::aten::ScalarType;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;

namespace parakeet {

namespace {

cv::Mat scale_with_padding(
    const cv::Mat& source,
    int* pad_x,
    int* pad_y,
    float* scale,
    cv::Size target_size) {
  int col = source.cols;
  int row = source.rows;

  if (col == target_size.width && row == target_size.height) {
    *pad_x = 0;
    *pad_y = 0;
    *scale = 1.0f;
    return source;
  }

  *scale = std::min(
      target_size.width / static_cast<float>(col),
      target_size.height / static_cast<float>(row));
  int resized_w = static_cast<int>(col * (*scale));
  int resized_h = static_cast<int>(row * (*scale));
  *pad_x = (target_size.width - resized_w) / 2;
  *pad_y = (target_size.height - resized_h) / 2;

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_w, resized_h));
  cv::Mat result =
      cv::Mat::zeros(target_size.height, target_size.width, source.type());
  resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));

  return result;
}

} // namespace

YOLODetector::YOLODetector(
    const std::string& model_path,
    const YOLOConfig& config,
    const std::string& data_path)
    : config_(config), is_initialized_(false) {
  if (!data_path.empty()) {
    model_ = std::make_unique<Module>(
        model_path, data_path, Module::LoadMode::Mmap);
  } else {
    model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  }
}

YOLODetector::~YOLODetector() = default;

bool YOLODetector::initialize() {
  if (is_initialized_) {
    return true;
  }

  auto error = model_->load();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load YOLO model");
    return false;
  }

  error = model_->load_forward();
  if (error != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method");
    return false;
  }

  // Get model input dimensions
  auto method_meta_result = model_->method_meta("forward");
  if (!method_meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata");
    return false;
  }

  auto input_meta_result = method_meta_result->input_tensor_meta(0);
  if (input_meta_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get input tensor metadata");
    return false;
  }

  auto sizes = input_meta_result->sizes();
  if (sizes.size() < 4) {
    ET_LOG(Error, "Invalid input tensor dimensions");
    return false;
  }

  // Assuming NCHW format: [batch, channels, height, width]
  input_size_ = cv::Size(sizes[3], sizes[2]);

  ET_LOG(
      Info,
      "YOLO detector initialized: input_size=%dx%d",
      input_size_.width,
      input_size_.height);

  is_initialized_ = true;
  return true;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame) {
  std::vector<Detection> detections;

  if (!is_initialized_) {
    ET_LOG(Error, "Detector not initialized");
    return detections;
  }

  if (frame.empty()) {
    return detections;
  }

  // Preprocess image
  int pad_x, pad_y;
  float scale;
  cv::Mat processed = scale_with_padding(frame, &pad_x, &pad_y, &scale, input_size_);

  // Convert to blob (NCHW format, normalized to [0, 1])
  cv::Mat blob;
  cv::dnn::blobFromImage(
      processed, blob, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);

  // Run inference
  auto input_tensor = from_blob(
      static_cast<void*>(blob.data),
      std::vector<int>(blob.size.p, blob.size.p + blob.dims),
      ScalarType::Float);

  auto result = model_->forward(input_tensor);
  if (!result.ok()) {
    ET_LOG(Error, "YOLO inference failed");
    return detections;
  }

  auto output_tensor = result->at(0).toTensor();

  // Parse YOLO output: shape is [batch, num_classes + 4, num_detections]
  // Format: [x, y, w, h, class_scores...]
  cv::Mat output_mat(
      output_tensor.dim() - 1,
      output_tensor.sizes().data() + 1,
      CV_32FC1,
      output_tensor.mutable_data_ptr());

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // Iterate over detections
  for (int i = 0; i < output_mat.cols; ++i) {
    // Extract class scores (skip first 4 values: x, y, w, h)
    const cv::Mat class_scores = output_mat.col(i).rowRange(4, output_mat.rows);

    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(class_scores, nullptr, &score, nullptr, &class_id_point);

    if (score <= config_.score_threshold) {
      continue;
    }

    class_ids.push_back(class_id_point.y);
    confidences.push_back(static_cast<float>(score));

    // Extract bounding box
    const float x = output_mat.at<float>(0, i);
    const float y = output_mat.at<float>(1, i);
    const float w = output_mat.at<float>(2, i);
    const float h = output_mat.at<float>(3, i);

    // Convert from center format to corner format and scale back to original
    const int left = static_cast<int>((x - 0.5f * w - pad_x) / scale);
    const int top = static_cast<int>((y - 0.5f * h - pad_y) / scale);
    const int width = static_cast<int>(w / scale);
    const int height = static_cast<int>(h / scale);

    boxes.push_back(cv::Rect(left, top, width, height));
  }

  // Apply Non-Maximum Suppression
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(
      boxes,
      confidences,
      config_.score_threshold,
      config_.nms_threshold,
      nms_indices);

  // Build final detections
  for (int idx : nms_indices) {
    Detection det;
    det.class_id = class_ids[idx];
    det.confidence = confidences[idx];
    det.box = boxes[idx];

    if (det.class_id < static_cast<int>(config_.class_names.size())) {
      det.class_name = config_.class_names[det.class_id];
    } else {
      det.class_name = "class_" + std::to_string(det.class_id);
    }

    detections.push_back(det);
  }

  return detections;
}

void YOLODetector::draw_detections(
    cv::Mat& frame,
    const std::vector<Detection>& detections,
    const cv::Scalar& color) {
  for (const auto& det : detections) {
    // Draw bounding box
    cv::rectangle(frame, det.box, color, 2);

    // Prepare label
    std::string label = det.class_name + " " +
        std::to_string(det.confidence).substr(0, 4);

    // Calculate text size and background box
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

    int text_x = det.box.x;
    int text_y = det.box.y - 10;
    if (text_y < text_size.height) {
      text_y = det.box.y + text_size.height + 10;
    }

    // Draw background for text
    cv::rectangle(
        frame,
        cv::Point(text_x, text_y - text_size.height - baseline),
        cv::Point(text_x + text_size.width, text_y + baseline),
        color,
        cv::FILLED);

    // Draw text
    cv::putText(
        frame,
        label,
        cv::Point(text_x, text_y - baseline),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(255, 255, 255),
        2);
  }
}

std::vector<std::string> YOLODetector::get_coco_classes() {
  return {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
}

} // namespace parakeet
