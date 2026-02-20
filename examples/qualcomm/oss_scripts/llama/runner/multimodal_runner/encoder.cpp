/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <fstream>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

EncoderRunner::EncoderRunner(const std::string& model_path)
    : image_seq_len_(0) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  ET_LOG(Info, "Creating encoder module: model_path=%s", model_path.c_str());
}

bool EncoderRunner::is_method_loaded() const {
  return module_->is_method_loaded(kEncoderForwardName);
}

Error EncoderRunner::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }

  auto load_result = module_->load_method(kEncoderForwardName);
  if (load_result != Error::Ok) {
    ET_LOG(Error, "Failed to load encoder method");
    return load_result;
  }

  // Get image sequence length from output metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  if (!method_meta.ok()) {
    ET_LOG(Error, "Failed to get encoder method metadata");
    return method_meta.error();
  }

  // vision embedding output shape: [1, seq_len, dim]
  image_seq_len_ = method_meta->output_tensor_meta(0)->sizes()[1];
  ET_LOG(Info, "Encoder loaded successfully, image_seq_len=%d", image_seq_len_);

  return Error::Ok;
}

int32_t EncoderRunner::get_image_seq_len() const {
  return image_seq_len_;
}

Result<Tensor> EncoderRunner::encode(TensorPtr& image_tensor) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  auto tensor_ptr = image_tensor.get();
  ET_LOG(Info, "Encoding image tensor with numel: %zu", tensor_ptr->numel());

  std::vector<executorch::runtime::EValue> encoder_inputs;
  encoder_inputs.emplace_back(*tensor_ptr);

  auto encoder_result = module_->forward(encoder_inputs);
  ET_CHECK_MSG(encoder_result.ok(), "Encoder execution failed");

  auto encoder_output = encoder_result.get();
  auto image_hidden_states = encoder_output[0].toTensor();
  ET_LOG(Info, "Encoder execution completed, got image hidden states");

  return image_hidden_states;
}

Result<Tensor> EncoderRunner::encode_from_file(
    const std::string& image_file_path) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  // Get input tensor metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  auto sizes_span = method_meta->input_tensor_meta(0)->sizes();

  // Calculate total number of elements
  int64_t num_elem = 1;
  for (const auto& size : sizes_span) {
    num_elem *= size;
  }

  // Read image data from file
  ET_LOG(
      Info,
      "Reading image from file: %s, num_elements=%ld",
      image_file_path.c_str(),
      num_elem);
  std::ifstream file(image_file_path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(
      file.is_open(), "Failed to open image file: %s", image_file_path.c_str());

  // To prevent users from passing images that have not been
  // resized to match the encoder input size.
  std::streamsize file_size = file.tellg();
  std::streamsize expected_size = num_elem * sizeof(float);
  ET_CHECK_MSG(
      file_size == expected_size,
      "Image file size mismatch: expected %ld bytes but got %ld bytes (file: %s)",
      expected_size,
      file_size,
      image_file_path.c_str());

  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(num_elem);
  file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
  file.close();

  // Create tensor from buffer
  TensorPtr tensor = executorch::extension::from_blob(
      buffer.data(),
      std::vector<int32_t>(sizes_span.begin(), sizes_span.end()),
      executorch::aten::ScalarType::Float);

  // Encode the tensor
  return encode(tensor);
}

} // namespace example
