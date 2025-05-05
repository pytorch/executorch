/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple mimi decoder runner that takes encoder's result as input.

#include <executorch/examples/qualcomm/oss_scripts/moshi/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>

#include <ctime>
#include <fstream>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

Runner::Runner(const std::string& model_path, const std::string& output_path)
    : output_path_(output_path) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  ET_LOG(Info, "creating module: model_path=%s", model_path.c_str());
}

Error Runner::parse_input_list(std::string& path) {
  // Fill in data for input
  std::ifstream input_list(path);
  ET_CHECK_MSG(input_list.is_open(), "Input list error opening file");
  std::string encoded_input;

  while (std::getline(input_list, encoded_input)) {
    std::ifstream is;
    is.open(encoded_input, std::ios::binary);
    is.seekg(0, std::ios::end);
    size_t filesize = is.tellg();
    is.seekg(0, std::ios::beg);
    std::vector<int32_t> encoded_in;
    encoded_in.resize(filesize / sizeof(int32_t));
    is.read(reinterpret_cast<char*>(encoded_in.data()), filesize);
    encoded_input_list_.first.push_back(encoded_in);
  }
  return Error::Ok;
}

Error Runner::init_io() {
  Result<MethodMeta> method_meta = module_->method_meta(method_name_);
  k_cache_.first.resize(
      (method_meta->input_tensor_meta(1)->nbytes() / sizeof(float)), 0);
  v_cache_.first.resize(
      (method_meta->input_tensor_meta(2)->nbytes() / sizeof(float)), 0);
  end_index_.first.resize(
      (method_meta->input_tensor_meta(3)->nbytes() / sizeof(float)), 0);
  end_offset_.first.resize(
      (method_meta->input_tensor_meta(4)->nbytes() / sizeof(float)), 0);

  for (int i = 5; i < 10; i++) {
    auto size = (method_meta->input_tensor_meta(i)->nbytes() / sizeof(float));
    std::vector<float> convtr_partial(size, 0);
    convtr_partials_.emplace_back(convtr_partial, nullptr);
  }

  for (int i = 10; i < method_meta->num_inputs(); i++) {
    auto size = (method_meta->input_tensor_meta(i)->nbytes() / sizeof(float));
    std::vector<float> conv_prev(size, 0);
    conv_previous_.emplace_back(conv_prev, nullptr);
  }

  auto per_decode_output_size =
      (method_meta->output_tensor_meta(0)->nbytes() / sizeof(float));
  for (int i = 0; i < encoded_input_list_.first.size(); i++) {
    std::vector<float> output(per_decode_output_size, 0);
    decoded_output_list_.first.emplace_back(output);
  }
  return Error::Ok;
}

Error Runner::prepare_io() {
  Result<MethodMeta> method_meta = module_->method_meta(method_name_);

  // in[0]
  Result<TensorInfo> encoded_input_meta = method_meta->input_tensor_meta(0);
  encoded_input_list_.second = std::make_shared<TensorImpl>(
      encoded_input_meta->scalar_type(),
      encoded_input_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(encoded_input_meta->sizes().data()),
      encoded_input_list_.first[0].data(),
      const_cast<TensorImpl::DimOrderType*>(
          encoded_input_meta->dim_order().data()));
  input_tensors_.emplace_back(encoded_input_list_.second.get());

  // out[0]
  Result<TensorInfo> decoded_output_meta = method_meta->output_tensor_meta(0);
  decoded_output_list_.second = std::make_shared<TensorImpl>(
      decoded_output_meta->scalar_type(),
      decoded_output_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(decoded_output_meta->sizes().data()),
      decoded_output_list_.first[0].data(),
      const_cast<TensorImpl::DimOrderType*>(
          decoded_output_meta->dim_order().data()));
  output_tensors_.emplace_back(decoded_output_list_.second.get());

  // in[1] and out[1]
  Result<TensorInfo> k_cache_meta = method_meta->input_tensor_meta(1);
  k_cache_.second = std::make_shared<TensorImpl>(
      k_cache_meta->scalar_type(),
      k_cache_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(k_cache_meta->sizes().data()),
      k_cache_.first.data(),
      const_cast<TensorImpl::DimOrderType*>(k_cache_meta->dim_order().data()));
  input_tensors_.emplace_back(k_cache_.second.get());
  output_tensors_.emplace_back(k_cache_.second.get());

  // in[2] and out[2]
  Result<TensorInfo> v_cache_meta = method_meta->input_tensor_meta(2);
  v_cache_.second = std::make_shared<TensorImpl>(
      v_cache_meta->scalar_type(),
      v_cache_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(v_cache_meta->sizes().data()),
      v_cache_.first.data(),
      const_cast<TensorImpl::DimOrderType*>(v_cache_meta->dim_order().data()));
  input_tensors_.emplace_back(v_cache_.second.get());
  output_tensors_.emplace_back(v_cache_.second.get());

  // in[3] and out[3]
  Result<TensorInfo> end_index_meta = method_meta->input_tensor_meta(3);
  end_index_.second = std::make_shared<TensorImpl>(
      end_index_meta->scalar_type(),
      end_index_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(end_index_meta->sizes().data()),
      end_index_.first.data(),
      const_cast<TensorImpl::DimOrderType*>(
          end_index_meta->dim_order().data()));
  input_tensors_.emplace_back(end_index_.second.get());
  output_tensors_.emplace_back(end_index_.second.get());

  // in[4] and out[4]
  Result<TensorInfo> end_offset_meta = method_meta->input_tensor_meta(4);
  end_offset_.second = std::make_shared<TensorImpl>(
      end_offset_meta->scalar_type(),
      end_offset_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(end_offset_meta->sizes().data()),
      end_offset_.first.data(),
      const_cast<TensorImpl::DimOrderType*>(
          end_offset_meta->dim_order().data()));
  input_tensors_.emplace_back(end_offset_.second.get());
  output_tensors_.emplace_back(end_offset_.second.get());

  // in[5-9] and out [5-9]
  for (int i = 0, convtr_partials_start = 5; i < convtr_partials_.size();
       i++, convtr_partials_start++) {
    Result<TensorInfo> convtr_partial_meta =
        method_meta->input_tensor_meta(convtr_partials_start);
    convtr_partials_[i].second = std::make_shared<TensorImpl>(
        convtr_partial_meta->scalar_type(),
        convtr_partial_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(convtr_partial_meta->sizes().data()),
        convtr_partials_[i].first.data(),
        const_cast<TensorImpl::DimOrderType*>(
            convtr_partial_meta->dim_order().data()));
    input_tensors_.emplace_back(convtr_partials_[i].second.get());
    output_tensors_.emplace_back(convtr_partials_[i].second.get());
  }

  // in[10-15] and out [10-15]
  for (int i = 0, conv_previous_start = 10; i < conv_previous_.size();
       i++, conv_previous_start++) {
    Result<TensorInfo> conv_previous_meta =
        method_meta->input_tensor_meta(conv_previous_start);
    conv_previous_[i].second = std::make_shared<TensorImpl>(
        conv_previous_meta->scalar_type(),
        conv_previous_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(conv_previous_meta->sizes().data()),
        conv_previous_[i].first.data(),
        const_cast<TensorImpl::DimOrderType*>(
            conv_previous_meta->dim_order().data()));
    input_tensors_.emplace_back(conv_previous_[i].second.get());
    output_tensors_.emplace_back(conv_previous_[i].second.get());
  }

  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }

  for (int i = 0; i < output_tensors_.size(); i++) {
    ET_CHECK_MSG(
        module_->set_output(method_name_, output_tensors_[i], i) == Error::Ok,
        "failed to set output tensor for module %u'th output",
        i);
  }

  return Error::Ok;
}

Error Runner::load(std::string& input_list) {
  if (module_->is_loaded()) {
    return Error::Ok;
  }
  method_name_ = *module_->method_names()->begin();
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(method_name_));
  ET_CHECK_OK_OR_RETURN_ERROR(parse_input_list(input_list));
  ET_CHECK_OK_OR_RETURN_ERROR(init_io());
  ET_CHECK_OK_OR_RETURN_ERROR(prepare_io());

  return Error::Ok;
}

Error Runner::generate(std::string& input_list) {
  std::vector<EValue> inputs;
  if (!module_->is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load(input_list));
    stats_.model_load_end_ms = time_in_ms();
  }

  ET_LOG(Info, "Start generating");
  stats_.decode_start_ms = time_in_ms();
  for (int i = 0; i < encoded_input_list_.first.size(); i++) {
    // encoded_input_list_ stores all executions' inputs. During each execution,
    // it only needs to update the pointer to the new input's address. Same for
    // decoded_output_list_ where memory space is reserved for all outputs.
    // During each execution, point the output to corresponding output memory
    // address. After exit for loop, dump all the outputs to a raw file.
    encoded_input_list_.second->set_data(encoded_input_list_.first[i].data());
    decoded_output_list_.second->set_data(decoded_output_list_.first[i].data());
    ET_CHECK_MSG(
        module_->set_output(method_name_, output_tensors_[0], 0) == Error::Ok,
        "failed to set output tensor for module 0'th output");

    module_->execute(method_name_, inputs_);
  }
  stats_.decode_end_ms = time_in_ms();

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t\t\t%ld (ms)",
      stats_.model_load_end_ms - stats_.model_load_start_ms);

  auto decode_duration = stats_.decode_end_ms - stats_.decode_start_ms;
  ET_LOG(
      Info,
      "\tTotal inference time for %zu chunks:\t\t%ld (ms)",
      encoded_input_list_.first.size(),
      decode_duration);

  ET_LOG(
      Info,
      "\tAverage inference time per chunk:\t\t%f (ms)",
      ((double)decode_duration / encoded_input_list_.first.size()));

  auto output_file_name = output_path_ + "/output_0_0.raw";
  std::ofstream fout(output_file_name.c_str(), std::ios::binary);
  for (const auto& decoded_output : decoded_output_list_.first) {
    fout.write(
        reinterpret_cast<const char*>(decoded_output.data()),
        decoded_output.size() * sizeof(float));
  }
  fout.close();

  return Error::Ok;
}

} // namespace example
