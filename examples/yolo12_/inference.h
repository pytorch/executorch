#ifndef INFERENCE_H
#define INFERENCE_H

#include <getopt.h>
#include <iostream>
#include <random>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <opencv2/opencv.hpp>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

struct Detection {
  int class_id{0};
  std::string className{};
  float confidence{0.0};
  cv::Scalar color{};
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
    const DetectionConfig yolo_config) {
  const auto model_input_shape =
      module.method_meta("forward")->input_tensor_meta(0)->sizes();
  cv::Size img_dims = {model_input_shape[2], model_input_shape[3]};

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

  ET_CHECK_MSG(result.ok(), "Could not infer the model with an error");

  const auto t = result->at(0).toTensor(); // Using only the 0 output
  cv::Mat mat_output(t.dim(), t.sizes().data(), CV_32FC1, t.data_ptr());

  // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes +
  // box[x,y,w,h])
  int rows = mat_output.size[2];
  int dimensions = mat_output.size[1];

  mat_output = mat_output.reshape(1, dimensions);
  cv::transpose(mat_output, mat_output);

  float* data = (float*)mat_output.data;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i) {
    float* classes_scores = data + 4;

    cv::Mat scores(1, yolo_config.classes.size(), CV_32FC1, classes_scores);
    cv::Point class_id;
    double max_class_score;

    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

    if (max_class_score > yolo_config.modelScoreThreshold) {
      confidences.push_back(max_class_score);
      class_ids.push_back(class_id.x);

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      int left = int((x - 0.5 * w - pad_x) / scale);
      int top = int((y - 0.5 * h - pad_y) / scale);

      int width = int(w / scale);
      int height = int(h / scale);

      boxes.push_back(cv::Rect(left, top, width, height));
    }

    data += dimensions;
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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

    result.className = yolo_config.classes[result.class_id];
    result.box = boxes[idx];

    detections.push_back(result);
  }

  return detections;
}
#endif // INFERENCE_H
