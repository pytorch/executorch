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
    cv::Mat& source,
    int* pad_x,
    int* pad_y,
    float* scale,
    cv::Size img_dims) {
  int col = source.cols;
  int row = source.rows;
  int m_inputWidth = img_dims.width;
  int m_inputHeight = img_dims.height;
  if (col == m_inputWidth and row == m_inputHeight) {
    return source;
  }

  *scale = std::min(m_inputWidth / (float)col, m_inputHeight / (float)row);
  int resized_w = col * *scale;
  int resized_h = row * *scale;
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
    const DetectionConfig yolo_config) {
  int pad_x, pad_y;
  float scale;
  input = scale_with_padding(input, &pad_x, &pad_y, &scale, img_dims);

  cv::Mat blob;
  cv::dnn::blobFromImage(
      input, blob, 1.0 / 255.0, img_dims, cv::Scalar(), true, false);
  const auto t_input = from_blob(
      (void*)blob.data,
      std::vector<int>(blob.size.p, blob.size.p + blob.dims),
      ScalarType::Float);
  const auto result = module.forward(t_input);

  ET_CHECK_MSG(
      result.ok(),
      "Execution of method forward failed with status 0x%" PRIx32,
      (uint32_t)result.error());

  const auto t = result->at(0).toTensor(); // Using only the 0 output
  // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes +
  // box[x,y,w,h])
  cv::Mat mat_output(t.dim() - 1, t.sizes().data() + 1, CV_32FC1, t.data_ptr());

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // Iterate over detections and collect class IDs, confidence scores, and
  // bounding boxes
  for (int i = 0; i < mat_output.cols; ++i) {
    const cv::Mat classes_scores =
        mat_output.col(i).rowRange(4, mat_output.rows);

    cv::Point class_id;
    double score;
    cv::minMaxLoc(
        classes_scores,
        nullptr,
        &score,
        nullptr,
        &class_id); // Find the class with the highest score

    // Check if the detection meets the confidence threshold
    if (score <= yolo_config.modelScoreThreshold)
      continue;

    class_ids.push_back(class_id.y);
    confidences.push_back(score);

    const float x = mat_output.at<float>(0, i);
    const float y = mat_output.at<float>(1, i);
    const float w = mat_output.at<float>(2, i);
    const float h = mat_output.at<float>(3, i);

    const int left = int((x - 0.5 * w - pad_x) / scale);
    const int top = int((y - 0.5 * h - pad_y) / scale);
    const int width = int(w / scale);
    const int height = int(h / scale);

    boxes.push_back(cv::Rect(left, top, width, height));
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(
      boxes,
      confidences,
      yolo_config.modelScoreThreshold,
      yolo_config.modelNMSThreshold,
      nms_result);

  std::vector<Detection> detections{};
  for (auto& idx : nms_result) {
    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];

    result.className = yolo_config.classes[result.class_id];
    result.box = boxes[idx];

    detections.push_back(result);
  }

  return detections;
}
#endif // INFERENCE_H
