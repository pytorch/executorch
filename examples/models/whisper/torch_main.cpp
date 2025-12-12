/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <tokenizers/tokenizers.h>
#include <torch/torch.h>

#include <executorch/extension/asr/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

#include "torch_whisper_runner.h"

DEFINE_string(
    encoder_library,
    "",
    "Path to the Whisper encoder AOTI shared library.");
DEFINE_string(
    decoder_library,
    "",
    "Path to the Whisper decoder AOTI shared library.");
DEFINE_string(
    encoder_weights,
    "",
    "Optional path to the encoder weights blob (.ptd).");
DEFINE_string(
    decoder_weights,
    "",
    "Optional path to the decoder weights blob (.ptd).");
DEFINE_string(
    tokenizer_path,
    ".",
    "Directory containing tokenizer.json, tokenizer_config.json, and special_tokens_map.json.");
DEFINE_string(
    processor_path,
    "",
    "Path to ExecuTorch preprocessor .pte for converting raw audio.");
DEFINE_string(audio_path, "", "Path to input audio file (.wav or raw .bin).");
DEFINE_string(
    model_name,
    "base",
    "Whisper model variant (base, small, medium, large, large-v2, large-v3, large-v3-turbo).");
DEFINE_double(
    temperature,
    0.0,
    "Sampling temperature. 0.0 performs greedy decoding.");
DEFINE_int32(max_new_tokens, 128, "Maximum number of tokens to generate.");
DEFINE_string(
    device,
    "cpu",
    "Device string passed to AOTI loader (e.g. cpu, cuda, cuda:0).");

namespace {

using ::executorch::extension::Module;
using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;

std::optional<std::string> as_optional(const std::string& path) {
  if (path.empty()) {
    return std::nullopt;
  }
  return path;
}

torch::ScalarType to_torch_dtype(::executorch::aten::ScalarType type) {
  switch (type) {
    case ::executorch::aten::ScalarType::Float:
      return torch::kFloat32;
    case ::executorch::aten::ScalarType::BFloat16:
      return torch::kBFloat16;
    case ::executorch::aten::ScalarType::Half:
      return torch::kFloat16;
    case ::executorch::aten::ScalarType::Long:
      return torch::kInt64;
    case ::executorch::aten::ScalarType::Int:
      return torch::kInt32;
    default:
      ET_LOG(
          Error,
          "Unsupported ExecuTorch dtype %d for conversion",
          static_cast<int>(type));
      return torch::kFloat32;
  }
}

torch::Tensor to_torch_tensor(const ::executorch::aten::Tensor& tensor) {
  std::vector<int64_t> sizes;
  sizes.reserve(tensor.dim());
  for (auto size : tensor.sizes()) {
    sizes.push_back(static_cast<int64_t>(size));
  }
  auto dtype = to_torch_dtype(tensor.scalar_type());
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  auto torch_tensor = torch::empty(sizes, options);
  const size_t num_bytes =
      tensor.numel() *
      ::executorch::runtime::elementSize(tensor.scalar_type());
  std::memcpy(
      torch_tensor.data_ptr(),
      tensor.mutable_data_ptr<void>(),
      num_bytes);
  return torch_tensor;
}

::executorch::runtime::Result<torch::Tensor> preprocess_audio() {
  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return Error::InvalidArgument;
  }
  if (FLAGS_processor_path.empty()) {
    ET_LOG(Error, "processor_path flag must be provided.");
    return Error::InvalidArgument;
  }

  auto audio_data =
      executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);

  Module processor(FLAGS_processor_path, Module::LoadMode::Mmap);
  auto load_status = processor.load();
  if (load_status != Error::Ok) {
    return load_status;
  }
  auto processed_result = processor.execute("forward", audio_tensor);
  if (processed_result.error() != Error::Ok) {
    return processed_result.error();
  }
  auto outputs = std::move(processed_result.get());
  if (outputs.empty() || !outputs[0].isTensor()) {
    ET_LOG(Error, "Preprocessor returned no tensor output.");
    return Error::Internal;
  }
  auto et_tensor = outputs[0].toTensor();
  return to_torch_tensor(et_tensor);
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_encoder_library.empty() || FLAGS_decoder_library.empty()) {
    ET_LOG(Error, "Both --encoder_library and --decoder_library are required.");
    return 1;
  }

  auto preprocess_result = preprocess_audio();
  if (!preprocess_result.ok()) {
    ET_LOG(
        Error,
        "Audio preprocessing failed with error code %d",
        static_cast<int>(preprocess_result.error()));
    return 1;
  }
  torch::Tensor features = std::move(preprocess_result.get());

  executorch::extension::asr::AsrTranscribeConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);
  if (FLAGS_model_name == "large-v2" || FLAGS_model_name == "large-v3" ||
      FLAGS_model_name == "large-v3-turbo") {
    config.decoder_start_token_id = 50258;
  } else {
    config.decoder_start_token_id = 50257;
  }

  TorchWhisperRunner runner(
      FLAGS_encoder_library,
      FLAGS_decoder_library,
      FLAGS_tokenizer_path,
      as_optional(FLAGS_encoder_weights),
      as_optional(FLAGS_decoder_weights),
      FLAGS_device);

  auto result = runner.transcribe(
      features,
      config,
      [&](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });

  if (!result.ok()) {
    ET_LOG(
        Error,
        "Transcription failed with error code %d",
        static_cast<int>(result.error()));
    return 1;
  }

  return 0;
}
