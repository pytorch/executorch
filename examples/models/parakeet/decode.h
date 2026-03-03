/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "types.h"

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace parakeet {

// TDT duration values
inline const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

// Helper to get expected scalar type for a method input
inline ::executorch::runtime::Result<::executorch::aten::ScalarType>
get_input_scalar_type(
    ::executorch::extension::Module& model,
    const char* method_name,
    size_t input_index) {
  auto method_meta_result = model.method_meta(method_name);
  if (!method_meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata for %s", method_name);
    return method_meta_result.error();
  }
  auto method_meta = method_meta_result.get();
  if (method_meta.num_inputs() <= input_index) {
    ET_LOG(
        Error,
        "Method %s has %zu inputs, but requested index %zu",
        method_name,
        method_meta.num_inputs(),
        input_index);
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto input_meta_result = method_meta.input_tensor_meta(input_index);
  if (input_meta_result.error() != ::executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to get input tensor metadata for %s[%zu]",
        method_name,
        input_index);
    return input_meta_result.error();
  }
  return input_meta_result.get().scalar_type();
}

// Greedy TDT decoding using the ExecuTorch Module API.
// Implements the label-looping variant of TDT greedy search.
inline std::vector<Token> greedy_decode_executorch(
    ::executorch::extension::Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10,
    ::executorch::extension::llm::Stats* stats = nullptr) {
  using ::executorch::extension::from_blob;

  std::vector<Token> hypothesis;

  // Shape: [1, T, joint_hidden]
  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  // Get expected dtype for decoder_step h and c inputs (indices 1 and 2)
  auto h_dtype_result = get_input_scalar_type(model, "decoder_step", 1);
  if (!h_dtype_result.ok()) {
    return hypothesis;
  }
  auto c_dtype_result = get_input_scalar_type(model, "decoder_step", 2);
  if (!c_dtype_result.ok()) {
    return hypothesis;
  }
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  ET_LOG(
      Info,
      "Decoder h dtype: %s, c dtype: %s",
      ::executorch::runtime::toString(h_dtype),
      ::executorch::runtime::toString(c_dtype));

  // Calculate buffer sizes based on dtype
  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

  // Initialize LSTM state with zeros (using byte buffers for dtype flexibility)
  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      h_dtype);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      c_dtype);

  // Prime the decoder with SOS (= blank_id)
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_step",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_step (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  // Get expected dtype for joint inputs
  auto f_dtype_result = get_input_scalar_type(model, "joint", 0);
  if (!f_dtype_result.ok()) {
    return hypothesis;
  }
  auto g_dtype_result = get_input_scalar_type(model, "joint", 1);
  if (!g_dtype_result.ok()) {
    return hypothesis;
  }
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  ET_LOG(
      Info,
      "Joint f dtype: %s, g dtype: %s",
      ::executorch::runtime::toString(f_dtype),
      ::executorch::runtime::toString(g_dtype));

  size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);

  // Copy g_proj data for reuse
  size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  // Scan over encoder output
  while (t < encoder_len) {
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }

    int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur_idx =
        joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      if (hypothesis.empty() && stats) {
        stats->first_token_ms = ::executorch::extension::llm::time_in_ms();
      }
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = model.execute(
          "decoder_step",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_step failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      std::memcpy(h_data.data(), new_h.const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), new_c.const_data_ptr(), c_data.size());
      std::memcpy(
          g_proj_data.data(), new_g_proj.const_data_ptr(), g_proj_data.size());

      t += dur;

      if (dur == 0) {
        symbols_on_frame++;
        if (symbols_on_frame >= max_symbols_per_step) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  return hypothesis;
}

} // namespace parakeet
