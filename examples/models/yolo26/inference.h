#ifndef INFERENCE_H
#define INFERENCE_H

#include <iostream>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <opencv2/opencv.hpp>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Result;

struct Detection {
  int class_id{0};
  std::string className{};
  float confidence{0.0};
  cv::Rect box{};
};

struct DetectionConfig {
  std::vector<std::string> classes;
  float modelScoreThreshold;
  float modelNMSThreshold;
};

cv::Mat scale_with_padding(
    const cv::Mat& source,
    int* pad_x,
    int* pad_y,
    float* scale,
    cv::Size img_dims) {
  int col = source.cols;
  int row = source.rows;
  int m_inputWidth = img_dims.width;
  int m_inputHeight = img_dims.height;
  if (col == m_inputWidth && row == m_inputHeight) {
    *pad_x = 0;
    *pad_y = 0;
    *scale = 1.f;
    return source;
  }

  *scale = std::min(
      m_inputWidth / static_cast<float>(col),
      m_inputHeight / static_cast<float>(row));
  int resized_w = static_cast<int>(col * *scale);
  int resized_h = static_cast<int>(row * *scale);
  *pad_x = (m_inputWidth - resized_w) / 2;
  *pad_y = (m_inputHeight - resized_h) / 2;

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_w, resized_h));
  cv::Mat result = cv::Mat::zeros(m_inputHeight, m_inputWidth, source.type());
  resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));
  resized.release();
  return result;
}

std::vector<Detection> infer_yolo_once(
    Module& module,
    cv::Mat input,
    cv::Size img_dims,
    const DetectionConfig& yolo_config) {
  int pad_x, pad_y;
  float scale;
  input = scale_with_padding(input, &pad_x, &pad_y, &scale, img_dims);

  cv::Mat blob;
  cv::dnn::blobFromImage(
      input, blob, 1.0 / 255.0, img_dims, cv::Scalar(), true, false);
  const auto t_input = from_blob(
      static_cast<void*>(blob.data),
      std::vector<int>(blob.size.p, blob.size.p + blob.dims),
      ScalarType::Float);
  const auto result = module.forward(t_input);

  ET_CHECK_MSG(
      result.ok(),
      "Execution of method forward failed with status 0x%" PRIx32,
      static_cast<uint32_t>(result.error()));

  // Yolo26 end-to-end (post-NMS) output format: [1, N, 6]
  // Each detection row: [x1, y1, x2, y2, confidence, class_id]
  const auto t = result->at(0).toTensor();
  ET_CHECK_MSG(
      t.dim() == 3 && t.sizes()[2] == 6,
      "Unexpected output shape: expected [1, N, 6] (end-to-end post-NMS format)");

  const int64_t num_detections = t.sizes()[1];
  const int num_classes = static_cast<int>(yolo_config.classes.size());
  const float* data = static_cast<const float*>(t.const_data_ptr());
  std::vector<Detection> detections;
  for (int64_t i = 0; i < num_detections; ++i) {
    const float* det = data + i * 6;
    const float x1 = det[0];
    const float y1 = det[1];
    const float x2 = det[2];
    const float y2 = det[3];
    const float confidence = det[4];
    const int class_id = static_cast<int>(det[5]);

    if (confidence <= yolo_config.modelScoreThreshold)
      continue;

    if (class_id < 0 || class_id >= num_classes)
      continue;

    // Map coordinates back to original image space
    const int left = static_cast<int>((x1 - pad_x) / scale);
    const int top = static_cast<int>((y1 - pad_y) / scale);
    const int width = static_cast<int>((x2 - x1) / scale);
    const int height = static_cast<int>((y2 - y1) / scale);

    Detection detection;
    detection.class_id = class_id;
    detection.confidence = confidence;
    detection.className = yolo_config.classes[class_id];
    detection.box = cv::Rect(left, top, width, height);
    detections.push_back(detection);
  }

  return detections;
}
#endif // INFERENCE_H
