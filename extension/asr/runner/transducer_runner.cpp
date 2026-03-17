/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/asr/runner/transducer_runner.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::asr {

namespace {

constexpr const char* kDecoderStepMethodName = "decoder_step";
constexpr const char* kJointMethodName = "joint";

// Helper to query expected scalar type for a method input at a given index.
Result<::executorch::aten::ScalarType> get_input_scalar_type(
    Module& module,
    const char* method_name,
    size_t input_index) {
  auto method_meta_result = module.method_meta(method_name);
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
    return Error::InvalidArgument;
  }
  auto input_meta_result = method_meta.input_tensor_meta(input_index);
  if (input_meta_result.error() != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to get input tensor metadata for %s[%zu]",
        method_name,
        input_index);
    return input_meta_result.error();
  }
  return input_meta_result.get().scalar_type();
}

} // namespace

TransducerRunner::TransducerRunner(
    const std::string& module_path,
    const std::string& tokenizer_path,
    TransducerConfig config)
    : module_path_(module_path),
      tokenizer_path_(tokenizer_path),
      config_(std::move(config)) {
  module_ = std::make_unique<Module>(module_path_, Module::LoadMode::Mmap);
}

bool TransducerRunner::is_loaded() const {
  return module_ && decoder_step_method_loaded_ &&
      joint_method_loaded_ && tokenizer_ && tokenizer_->is_loaded();
}

Error TransducerRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  stats_.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  ET_CHECK_OR_RETURN_ERROR(
      module_ != nullptr,
      InvalidArgument,
      "Module handle is null for path %s",
      module_path_.c_str());

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load());

  auto method_names_result = module_->method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
  const auto& method_names = method_names_result.get();

  ET_CHECK_OR_RETURN_ERROR(
      method_names.count(kDecoderStepMethodName) &&
          method_names.count(kJointMethodName),
      InvalidArgument,
      "Required methods not found. decoder_step=%d, joint=%d",
      static_cast<int>(method_names.count(kDecoderStepMethodName)),
      static_cast<int>(method_names.count(kJointMethodName)));

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kDecoderStepMethodName));
  decoder_step_method_loaded_ = true;

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kJointMethodName));
  joint_method_loaded_ = true;

  // Load tokenizer
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path_);
  ET_CHECK_OR_RETURN_ERROR(
      tokenizer,
      Internal,
      "Failed to create tokenizer from %s",
      tokenizer_path_.c_str());
  tokenizer_ = std::move(tokenizer);

  stats_.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  return Error::Ok;
}

Result<std::vector<TransducerToken>> TransducerRunner::transcribe(
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    std::function<void(const std::string&)> token_callback) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  stats_.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  std::vector<TransducerToken> hypothesis;

  // Shape: [1, T, joint_hidden]
  size_t proj_dim = static_cast<size_t>(encoder_output.sizes()[2]);

  // Get expected dtypes for decoder_step h and c inputs (indices 1 and 2)
  auto h_dtype_result =
      get_input_scalar_type(*module_, kDecoderStepMethodName, 1);
  ET_CHECK_OK_OR_RETURN_ERROR(h_dtype_result.error());
  auto c_dtype_result =
      get_input_scalar_type(*module_, kDecoderStepMethodName, 2);
  ET_CHECK_OK_OR_RETURN_ERROR(c_dtype_result.error());
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  // Calculate buffer sizes based on dtype
  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements = static_cast<size_t>(config_.num_rnn_layers) *
      static_cast<size_t>(config_.pred_hidden);

  // Initialize LSTM state with zeros (using byte buffers for dtype flexibility)
  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = ::executorch::extension::from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(config_.num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(config_.pred_hidden)},
      h_dtype);
  auto c = ::executorch::extension::from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(config_.num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(config_.pred_hidden)},
      c_dtype);

  // Prime the decoder with SOS (= blank_id) to match NeMo TDT label-looping
  std::vector<int64_t> sos_token_data = {config_.blank_id};
  auto sos_token = ::executorch::extension::from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = module_->execute(
      kDecoderStepMethodName,
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_init_result.error());

  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  // Get expected dtypes for joint inputs (f and g at indices 0 and 1)
  auto f_dtype_result = get_input_scalar_type(*module_, kJointMethodName, 0);
  ET_CHECK_OK_OR_RETURN_ERROR(f_dtype_result.error());
  auto g_dtype_result = get_input_scalar_type(*module_, kJointMethodName, 1);
  ET_CHECK_OK_OR_RETURN_ERROR(g_dtype_result.error());
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

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
      static_cast<const uint8_t*>(encoder_output.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  // Use configured durations or default to standard RNN-T behavior
  const bool is_tdt = !config_.durations.empty();

  stats_.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();

  // Scan over encoder output frames
  while (t < encoder_len) {
    // Get encoder frame at time t: encoder_output[:, t:t+1, :]
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);

    auto f_t = ::executorch::extension::from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = ::executorch::extension::from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result = module_->execute(
        kJointMethodName,
        std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    ET_CHECK_OK_OR_RETURN_ERROR(joint_result.error());

    int64_t k =
        joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = 1; // default for standard RNN-T
    if (is_tdt) {
      int64_t dur_idx =
          joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
      dur = config_.durations[static_cast<size_t>(dur_idx)];
    }

    if (k == config_.blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      if (hypothesis.empty()) {
        stats_.first_token_ms = ::executorch::extension::llm::time_in_ms();
      }
      hypothesis.push_back(
          {static_cast<uint64_t>(k), t, dur});

      // Invoke token callback
      if (token_callback && tokenizer_) {
        uint64_t prev_tok = hypothesis.size() > 1
            ? hypothesis[hypothesis.size() - 2].id
            : static_cast<uint64_t>(config_.blank_id);
        auto piece_result =
            tokenizer_->decode(prev_tok, static_cast<uint64_t>(k));
        if (piece_result.ok()) {
          token_callback(piece_result.get());
        }
      }

      std::vector<int64_t> token_data = {k};
      auto token = ::executorch::extension::from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = module_->execute(
          kDecoderStepMethodName,
          std::vector<::executorch::runtime::EValue>{token, h, c});
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_result.error());

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
        if (symbols_on_frame >= config_.max_symbols_per_step) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  stats_.num_generated_tokens = static_cast<int64_t>(hypothesis.size());
  stats_.inference_end_ms = ::executorch::extension::llm::time_in_ms();
  ::executorch::extension::llm::print_report(stats_);

  return hypothesis;
}

} // namespace executorch::extension::asr
