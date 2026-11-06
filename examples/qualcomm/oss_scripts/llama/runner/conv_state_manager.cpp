/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/conv_state_manager.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <cinttypes>
#include <cstring>

using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

ConvStateManager::ConvStateManager(Metadata metadata) : metadata_(metadata) {
  states_.resize(metadata_.num_layers, nullptr);
  outputs_.resize(metadata_.num_layers, nullptr);
}

Result<std::unique_ptr<ConvStateManager>> ConvStateManager::from_module(
    executorch::extension::Module& module,
    const MethodMeta& method_meta,
    int32_t max_ar_len) {
  if (module.method_names()->count("get_n_conv_layers") == 0) {
    return std::unique_ptr<ConvStateManager>(nullptr);
  }
  auto n_conv = module.get("get_n_conv_layers");
  ET_CHECK_OK_OR_RETURN_ERROR(n_conv.error());
  int64_t num_layers = n_conv->toScalar().to<int64_t>();
  ET_CHECK_OR_RETURN_ERROR(
      num_layers > 0 &&
          num_layers < static_cast<int64_t>(method_meta.num_inputs()),
      InvalidProgram,
      "Invalid get_n_conv_layers: %" PRId64,
      num_layers);
  auto state =
      method_meta.input_tensor_meta(method_meta.num_inputs() - num_layers);
  ET_CHECK_OK_OR_RETURN_ERROR(state.error());
  ET_CHECK_OR_RETURN_ERROR(
      state->sizes().size() == 3 && state->sizes()[0] == 1,
      InvalidProgram,
      "Conv state input must be [1, state_len, dim]");
  return std::make_unique<ConvStateManager>(Metadata{
      num_layers,
      state->sizes()[1],
      state->sizes()[2],
      max_ar_len,
      executorch::runtime::elementSize(state->scalar_type())});
}

void ConvStateManager::init(IMemAlloc* buffer_manager) {
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    states_[layer] = buffer_manager->allocate(state_nbytes());
    outputs_[layer] = buffer_manager->allocate(output_nbytes());
  }
  reset();
}

void ConvStateManager::bind_io(
    IMemAlloc* buffer_manager,
    const MethodMeta& method_meta,
    std::vector<std::unique_ptr<TensorImpl>>& tensors,
    std::vector<executorch::aten::Tensor>& input_tensors,
    std::vector<executorch::aten::Tensor>& output_tensors) {
  const size_t first_input = method_meta.num_inputs() - metadata_.num_layers;
  const size_t first_output = method_meta.num_outputs() - metadata_.num_layers;
  ET_CHECK_MSG(
      input_tensors.size() == first_input &&
          output_tensors.size() == first_output,
      "Conv states must be the last %" PRId64
      " inputs/outputs: bound %zu/%zu, expected %zu/%zu",
      metadata_.num_layers,
      input_tensors.size(),
      output_tensors.size(),
      first_input,
      first_output);
  auto bind = [&](const Result<TensorInfo>& info,
                  std::byte* data,
                  size_t capacity) {
    ET_CHECK_MSG(info.ok(), "Failed to get conv state/output tensor meta");
    ET_CHECK_MSG(
        info->sizes().size() == 3 && info->sizes()[0] == 1 &&
            info->sizes()[2] == metadata_.dim &&
            executorch::runtime::elementSize(info->scalar_type()) ==
                metadata_.element_size &&
            info->nbytes() <= capacity,
        "Unexpected conv state/output tensor for this manager");
    tensors.emplace_back(
        std::make_unique<TensorImpl>(
            info->scalar_type(),
            info->sizes().size(),
            const_cast<TensorImpl::SizesType*>(info->sizes().data()),
            data,
            const_cast<TensorImpl::DimOrderType*>(info->dim_order().data())));
    buffer_manager->add_memory_info(data, tensors.back()->nbytes(), info.get());
    return executorch::aten::Tensor(tensors.back().get());
  };
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    auto info = method_meta.input_tensor_meta(first_input + layer);
    ET_CHECK_MSG(info.ok(), "Failed to get conv state input tensor meta");
    ET_CHECK_MSG(
        info->sizes()[1] == metadata_.state_len, "Conv state length mismatch");
    input_tensors.emplace_back(bind(info, states_[layer], state_nbytes()));
  }
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    output_tensors.emplace_back(bind(
        method_meta.output_tensor_meta(first_output + layer),
        outputs_[layer],
        output_nbytes()));
  }
}

void ConvStateManager::reset() {
  // Byte 0 is not necessarily the quantized 0.0; the graph masks state rows
  // that precede position 0, so this only clears stale data.
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    std::memset(states_[layer], 0, state_nbytes());
  }
}

void ConvStateManager::update(int32_t n_update) {
  const size_t row = row_nbytes();
  const int64_t state_len = metadata_.state_len;
  for (int64_t layer = 0; layer < metadata_.num_layers; ++layer) {
    std::byte* state = states_[layer];
    const std::byte* out = outputs_[layer];
    if (n_update >= state_len) {
      std::memcpy(state, out + (n_update - state_len) * row, state_nbytes());
    } else {
      std::memmove(state, state + n_update * row, (state_len - n_update) * row);
      std::memcpy(state + (state_len - n_update) * row, out, n_update * row);
    }
  }
}

} // namespace example
