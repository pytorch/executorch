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

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::asr {

namespace {

constexpr const char* kEncoderMethodName = "encoder";
constexpr const char* kDecoderStepMethodName = "decoder_step";
constexpr const char* kJointMethodName = "joint";
constexpr const char* kPreprocessorMethodName = "preprocessor";

// Helper to get expected scalar type for a method input.
::executorch::runtime::Result<::executorch::aten::ScalarType>
get_input_scalar_type(
    Module& model,
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
    TransducerConfig config,
    std::optional<std::string> data_path)
    : module_path_(module_path),
      data_path_(data_path.value_or("")),
      tokenizer_path_(tokenizer_path),
      config_(std::move(config)) {
  if (data_path_.empty()) {
    module_ = std::make_unique<Module>(module_path_, Module::LoadMode::Mmap);
  } else {
    module_ = std::make_unique<Module>(
        module_path_, data_path_, Module::LoadMode::Mmap);
  }
}

bool TransducerRunner::is_loaded() const {
  return module_ && encoder_method_loaded_ && decoder_method_loaded_ &&
      joint_method_loaded_ &&
      (!preprocessor_method_present_ || preprocessor_method_loaded_) &&
      tokenizer_ && tokenizer_->is_loaded();
}

Error TransducerRunner::load_tokenizer() {
  if (tokenizer_ && tokenizer_->is_loaded()) {
    return Error::Ok;
  }

  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path_);
  ET_CHECK_OR_RETURN_ERROR(
      tokenizer,
      Internal,
      "Failed to create tokenizer from %s",
      tokenizer_path_.c_str());

  tokenizer_ = std::move(tokenizer);
  if (!tokenizer_->is_loaded()) {
    ET_LOG(
        Error,
        "Tokenizer reported unloaded state after load: %s",
        tokenizer_path_.c_str());
    return Error::Internal;
  }
  return Error::Ok;
}

Error TransducerRunner::load_model_metadata() {
  std::vector<::executorch::runtime::EValue> empty_inputs;

  auto read_int_constant =
      [&](const char* name, int64_t& out) -> Error {
    auto result = module_->execute(name, empty_inputs);
    ET_CHECK_OR_RETURN_ERROR(
        result.ok(),
        Internal,
        "Model must export '%s' as a constant_method.",
        name);
    auto& outputs = result.get();
    ET_CHECK_OR_RETURN_ERROR(
        !outputs.empty() && outputs[0].isInt(),
        Internal,
        "constant_method '%s' returned %zu outputs; expected at least 1 int.",
        name,
        outputs.size());
    out = outputs[0].toInt();
    return Error::Ok;
  };

  ET_CHECK_OK_OR_RETURN_ERROR(read_int_constant("blank_id", blank_id_));
  ET_CHECK_OK_OR_RETURN_ERROR(
      read_int_constant("num_rnn_layers", num_rnn_layers_));
  ET_CHECK_OK_OR_RETURN_ERROR(read_int_constant("pred_hidden", pred_hidden_));

  ET_LOG(
      Info,
      "Model metadata: blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld",
      static_cast<long long>(blank_id_),
      static_cast<long long>(num_rnn_layers_),
      static_cast<long long>(pred_hidden_));

  return Error::Ok;
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
      method_names.count(kEncoderMethodName) &&
          method_names.count(kDecoderStepMethodName) &&
          method_names.count(kJointMethodName),
      InvalidArgument,
      "Required methods not found. encoder=%d, decoder_step=%d, joint=%d",
      static_cast<int>(method_names.count(kEncoderMethodName)),
      static_cast<int>(method_names.count(kDecoderStepMethodName)),
      static_cast<int>(method_names.count(kJointMethodName)));

  preprocessor_method_present_ = method_names.count(kPreprocessorMethodName);

  // Read model metadata from constant_methods
  ET_CHECK_OK_OR_RETURN_ERROR(load_model_metadata());

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kEncoderMethodName));
  encoder_method_loaded_ = true;

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kDecoderStepMethodName));
  decoder_method_loaded_ = true;

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kJointMethodName));
  joint_method_loaded_ = true;

  if (preprocessor_method_present_) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kPreprocessorMethodName));
    preprocessor_method_loaded_ = true;
  }

  ET_CHECK_OK_OR_RETURN_ERROR(load_tokenizer());

  stats_.model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  return Error::Ok;
}

Result<::executorch::extension::TensorPtr> TransducerRunner::preprocess(
    ::executorch::extension::TensorPtr raw_audio) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  ET_CHECK_OR_RETURN_ERROR(
      preprocessor_method_present_,
      InvalidState,
      "Model does not have a 'preprocessor' method. "
      "Provide preprocessed features directly to transcribe().");

  int64_t audio_len = raw_audio->numel();
  std::vector<int64_t> audio_len_data = {audio_len};
  auto audio_len_tensor = ::executorch::extension::from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto result = module_->execute(
      kPreprocessorMethodName,
      std::vector<::executorch::runtime::EValue>{raw_audio, audio_len_tensor});
  ET_CHECK_OK_OR_RETURN_ERROR(result.error());

  auto& outputs = result.get();
  ET_CHECK_OR_RETURN_ERROR(
      !outputs.empty() && outputs[0].isTensor(),
      Internal,
      "Preprocessor returned unexpected output.");

  auto mel = outputs[0].toTensor();
  return std::make_shared<::executorch::aten::Tensor>(std::move(mel));
}

Result<std::vector<Token>> TransducerRunner::transcribe(
    ::executorch::extension::TensorPtr preprocessed_features,
    std::function<void(const std::string&)> token_callback) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  stats_.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  // --- Encode ---
  int64_t mel_len_value = preprocessed_features->size(1);
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len = ::executorch::extension::from_blob(
      mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto enc_result = module_->execute(
      kEncoderMethodName,
      std::vector<::executorch::runtime::EValue>{
          preprocessed_features, mel_len});
  ET_CHECK_OK_OR_RETURN_ERROR(enc_result.error());

  auto& enc_outputs = enc_result.get();
  ET_CHECK_OR_RETURN_ERROR(
      enc_outputs.size() >= 2 && enc_outputs[0].isTensor() &&
          enc_outputs[1].isTensor(),
      Internal,
      "Encoder expected to return (f_proj, encoded_len), got %zu outputs.",
      enc_outputs.size());

  auto f_proj = enc_outputs[0].toTensor(); // [B, T, joint_hidden]
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  stats_.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
  stats_.num_prompt_tokens = encoded_len;

  ET_LOG(
      Info,
      "Encoder output shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(f_proj.sizes()[0]),
      static_cast<long>(f_proj.sizes()[1]),
      static_cast<long>(f_proj.sizes()[2]),
      static_cast<long>(encoded_len));

  // --- Prepare LSTM state ---
  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  auto h_dtype_result =
      get_input_scalar_type(*module_, kDecoderStepMethodName, 1);
  ET_CHECK_OK_OR_RETURN_ERROR(h_dtype_result.error());
  auto c_dtype_result =
      get_input_scalar_type(*module_, kDecoderStepMethodName, 2);
  ET_CHECK_OK_OR_RETURN_ERROR(c_dtype_result.error());
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers_) * static_cast<size_t>(pred_hidden_);

  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = ::executorch::extension::from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden_)},
      h_dtype);
  auto c = ::executorch::extension::from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden_)},
      c_dtype);

  // --- Prime decoder with SOS (= blank_id) ---
  std::vector<int64_t> sos_token_data = {blank_id_};
  auto sos_token = ::executorch::extension::from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = module_->execute(
      kDecoderStepMethodName,
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_init_result.error());

  auto& init_outputs = decoder_init_result.get();
  ET_CHECK_OR_RETURN_ERROR(
      init_outputs.size() >= 3 && init_outputs[0].isTensor() &&
          init_outputs[1].isTensor() && init_outputs[2].isTensor(),
      Internal,
      "Method %s expected to return (g_proj, h, c), got %zu outputs.",
      kDecoderStepMethodName,
      init_outputs.size());
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  // --- Query joint input dtypes ---
  auto f_dtype_result = get_input_scalar_type(*module_, kJointMethodName, 0);
  ET_CHECK_OK_OR_RETURN_ERROR(f_dtype_result.error());
  auto g_dtype_result = get_input_scalar_type(*module_, kJointMethodName, 1);
  ET_CHECK_OK_OR_RETURN_ERROR(g_dtype_result.error());
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);

  size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  // --- Greedy decode ---
  std::vector<Token> hypothesis;
  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  const bool is_tdt = !config_.durations.empty();
  uint64_t prev_token_id = static_cast<uint64_t>(blank_id_);

  std::vector<uint8_t> f_t_data(f_t_num_bytes);

  while (t < encoded_len) {
    // Extract encoder frame at time t
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

    auto& joint_outputs = joint_result.get();
    const size_t required_joint_outputs = is_tdt ? 2 : 1;
    ET_CHECK_OR_RETURN_ERROR(
        joint_outputs.size() >= required_joint_outputs &&
            joint_outputs[0].isTensor() &&
            (!is_tdt || joint_outputs[1].isTensor()),
        Internal,
        "Method %s expected to return %s, got %zu outputs.",
        kJointMethodName,
        is_tdt ? "(token_id, duration_idx)" : "(token_id,)",
        joint_outputs.size());

    int64_t k = joint_outputs[0].toTensor().const_data_ptr<int64_t>()[0];

    // Compute frame advance duration
    int64_t dur = 1; // default for standard RNN-T
    if (is_tdt) {
      int64_t dur_idx =
          joint_outputs[1].toTensor().const_data_ptr<int64_t>()[0];
      ET_CHECK_OR_RETURN_ERROR(
          dur_idx >= 0 &&
              static_cast<size_t>(dur_idx) < config_.durations.size(),
          Internal,
          "Joint network returned invalid duration index %lld (max %zu)",
          static_cast<long long>(dur_idx),
          config_.durations.size());
      dur = config_.durations[static_cast<size_t>(dur_idx)];
    }

    if (k == blank_id_) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      if (hypothesis.empty()) {
        stats_.first_token_ms = ::executorch::extension::llm::time_in_ms();
      }

      uint64_t token_id = static_cast<uint64_t>(k);
      hypothesis.push_back({token_id, t, dur});

      // Stream decoded text
      if (token_callback && tokenizer_) {
        auto piece_result = tokenizer_->decode(prev_token_id, token_id);
        if (piece_result.ok()) {
          token_callback(piece_result.get());
        } else {
          ET_LOG(
              Error,
              "Tokenizer failed to decode token pair (%llu, %llu)",
              static_cast<unsigned long long>(prev_token_id),
              static_cast<unsigned long long>(token_id));
        }
      }
      prev_token_id = token_id;

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = ::executorch::extension::from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = module_->execute(
          kDecoderStepMethodName,
          std::vector<::executorch::runtime::EValue>{token, h, c});
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_result.error());

      auto& outputs = decoder_result.get();
      ET_CHECK_OR_RETURN_ERROR(
          outputs.size() >= 3 && outputs[0].isTensor() &&
              outputs[1].isTensor() && outputs[2].isTensor(),
          Internal,
          "Method %s expected to return (g_proj, h, c), got %zu outputs.",
          kDecoderStepMethodName,
          outputs.size());
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
