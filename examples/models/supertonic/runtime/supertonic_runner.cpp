/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "supertonic_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

#ifndef SUPERTONIC_PURE_HELPERS_ONLY
#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <mlx/stream.h>
#endif

namespace supertonic {
namespace {

size_t element_count(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension <= 0 ||
        count > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(dimension)) {
      return 0;
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void validate_tensor(
    const TensorView& tensor,
    const std::string& name,
    const std::vector<int64_t>& expected_shape) {
  if (tensor.dtype != TensorDtype::Float16) {
    throw std::invalid_argument(name + " must have dtype float16");
  }
  if (!tensor.contiguous) {
    throw std::invalid_argument(name + " must be contiguous");
  }
  if (tensor.shape != expected_shape) {
    throw std::invalid_argument(name + " has an incompatible shape");
  }
  const size_t count = element_count(tensor.shape);
  if (tensor.values == nullptr || tensor.values->size() != count) {
    throw std::invalid_argument(name + " data does not match its shape");
  }
  for (float value : *tensor.values) {
    if (!std::isfinite(value) || std::abs(value) > 65504.0f) {
      throw std::invalid_argument(
          name + " values must be within the finite FP16 range");
    }
  }
}

bool has_valid_mask(const TensorView& tensor) {
  double sum = 0.0;
  for (float value : *tensor.values) {
    if (!std::isfinite(value)) {
      return false;
    }
    sum += value;
  }
  return std::isfinite(sum) && sum > 0.0;
}

int64_t checked_sample_count(
    float seconds,
    int64_t sample_rate,
    const char* label,
    bool allow_zero = false) {
  if (!std::isfinite(seconds) || seconds < 0.0f || sample_rate <= 0) {
    throw std::invalid_argument(std::string("invalid ") + label);
  }
  const long double count =
      static_cast<long double>(seconds) * static_cast<long double>(sample_rate);
  if (!std::isfinite(count) ||
      count > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(
        std::string(label) + " sample count is unrepresentable");
  }
  const int64_t result = static_cast<int64_t>(count);
  if (!allow_zero && result <= 0) {
    throw std::invalid_argument(std::string(label) + " produces no samples");
  }
  return result;
}

size_t checked_add_size(size_t first, size_t second, const char* label) {
  if (first > std::numeric_limits<size_t>::max() - second) {
    throw std::overflow_error(std::string(label) + " size is unrepresentable");
  }
  return first + second;
}

std::string joined_names(const std::set<std::string>& names) {
  std::string result;
  for (const auto& name : names) {
    result += (result.empty() ? "" : ", ") + name;
  }
  return result;
}

} // namespace

MetadataValue MetadataValue::integer(int64_t value) {
  MetadataValue result{};
  result.type = MetadataValueType::Integer;
  result.integer_value = value;
  return result;
}

MetadataValue MetadataValue::boolean(bool value) {
  MetadataValue result{};
  result.type = MetadataValueType::Boolean;
  result.boolean_value = value;
  return result;
}

MetadataValue MetadataValue::string(std::string value) {
  MetadataValue result{};
  result.type = MetadataValueType::String;
  result.string_value = std::move(value);
  return result;
}

RuntimeMetadata validate_metadata_contract(
    const std::set<std::string>& method_names,
    const std::map<std::string, MetadataValue>& metadata_values) {
  const std::set<std::string> expected_methods = {
      "duration_predictor",
      "text_encoder",
      "vector_estimator",
      "vocoder",
      "get_sample_rate",
      "get_base_chunk_size",
      "get_chunk_compress_factor",
      "get_flow_steps",
      "get_text_vocabulary_size",
      "get_latent_dim",
      "get_latent_channels",
      "get_max_text_length",
      "get_max_latent_length",
      "get_batch_size",
      "get_activation_dtype",
      "enable_dynamic_shape"};
  std::set<std::string> missing;
  std::set_difference(
      expected_methods.begin(),
      expected_methods.end(),
      method_names.begin(),
      method_names.end(),
      std::inserter(missing, missing.end()));
  if (!missing.empty()) {
    throw std::runtime_error("missing methods: " + joined_names(missing));
  }
  std::set<std::string> unexpected;
  std::set_difference(
      method_names.begin(),
      method_names.end(),
      expected_methods.begin(),
      expected_methods.end(),
      std::inserter(unexpected, unexpected.end()));
  if (!unexpected.empty()) {
    throw std::runtime_error("unexpected methods: " + joined_names(unexpected));
  }

  const auto require = [&](const char* name,
                           MetadataValueType type) -> const MetadataValue& {
    const auto found = metadata_values.find(name);
    if (found == metadata_values.end()) {
      throw std::runtime_error(std::string("missing metadata value: ") + name);
    }
    if (found->second.type != type) {
      const char* expected = type == MetadataValueType::Integer
          ? "an integer"
          : (type == MetadataValueType::Boolean ? "a boolean" : "a string");
      throw std::runtime_error(std::string(name) + " must be " + expected);
    }
    return found->second;
  };
  const auto integer = [&](const char* name) {
    return require(name, MetadataValueType::Integer).integer_value;
  };

  RuntimeMetadata metadata;
  metadata.sample_rate = integer("get_sample_rate");
  metadata.base_chunk_size = integer("get_base_chunk_size");
  metadata.chunk_compress_factor = integer("get_chunk_compress_factor");
  metadata.flow_steps = integer("get_flow_steps");
  metadata.text_vocabulary_size = integer("get_text_vocabulary_size");
  metadata.latent_dim = integer("get_latent_dim");
  metadata.latent_channels = integer("get_latent_channels");
  metadata.max_text_length = integer("get_max_text_length");
  metadata.max_latent_length = integer("get_max_latent_length");
  metadata.batch_size = integer("get_batch_size");
  metadata.activation_dtype =
      require("get_activation_dtype", MetadataValueType::String).string_value;
  metadata.dynamic_shapes =
      require("enable_dynamic_shape", MetadataValueType::Boolean).boolean_value;
  if (metadata.sample_rate != 44100 || metadata.base_chunk_size != 512 ||
      metadata.chunk_compress_factor != 6 || metadata.flow_steps != 5 ||
      metadata.text_vocabulary_size <= 0 || metadata.latent_dim != 24 ||
      metadata.latent_channels != 144 || metadata.batch_size != 1 ||
      metadata.activation_dtype != "float16" || !metadata.dynamic_shapes ||
      metadata.max_text_length < 2 || metadata.max_text_length > 1000 ||
      metadata.max_latent_length < 2 || metadata.max_latent_length > 1000) {
    throw std::runtime_error(
        "Supertonic PTE metadata is incompatible with the batch-1 FP16 "
        "44.1 kHz five-step runner");
  }
  if (metadata.latent_dim > std::numeric_limits<int64_t>::max() /
              metadata.chunk_compress_factor ||
      metadata.latent_channels !=
          metadata.latent_dim * metadata.chunk_compress_factor) {
    throw std::runtime_error("Supertonic latent metadata is inconsistent");
  }
  return metadata;
}

void validate_vector_inputs(
    const VectorInputs& inputs,
    const RuntimeMetadata& metadata) {
  const auto& latent_shape = inputs.noisy_latent.shape;
  const auto& text_shape = inputs.text_emb.shape;
  if (latent_shape.size() != 3 || text_shape.size() != 3) {
    throw std::invalid_argument(
        "noisy_latent and text_emb must be rank-three tensors");
  }
  const int64_t batch = latent_shape[0];
  if (batch != 1 || text_shape[0] != 1) {
    throw std::invalid_argument("vector_estimator batch size must be 1");
  }
  const int64_t latent_length = latent_shape[2];
  const int64_t text_length = text_shape[2];
  if (latent_length <= 0 || latent_length > metadata.max_latent_length) {
    throw std::invalid_argument("latent length is outside the exported bounds");
  }
  if (text_length <= 0 || text_length > metadata.max_text_length) {
    throw std::invalid_argument("text length is outside the exported bounds");
  }
  validate_tensor(
      inputs.noisy_latent,
      "noisy_latent",
      {1, metadata.latent_channels, latent_length});
  validate_tensor(inputs.text_emb, "text_emb", {1, 256, text_length});
  validate_tensor(inputs.style_ttl, "style_ttl", {1, 50, 256});
  if (inputs.latent_mask.shape != std::vector<int64_t>({1, 1, latent_length})) {
    throw std::invalid_argument(
        "noisy_latent and latent_mask latent lengths must match");
  }
  validate_tensor(inputs.latent_mask, "latent_mask", {1, 1, latent_length});
  if (inputs.text_mask.shape != std::vector<int64_t>({1, 1, text_length})) {
    throw std::invalid_argument(
        "text_emb and text_mask text lengths must match");
  }
  validate_tensor(inputs.text_mask, "text_mask", {1, 1, text_length});
  if (inputs.current_step.values != nullptr &&
      inputs.current_step.values->size() == 1 &&
      !std::isfinite(inputs.current_step.values->front())) {
    throw std::invalid_argument("current_step must be finite");
  }
  if (inputs.total_step.values != nullptr &&
      inputs.total_step.values->size() == 1 &&
      (!std::isfinite(inputs.total_step.values->front()) ||
       inputs.total_step.values->front() <= 0.0f)) {
    throw std::invalid_argument("total_step must be finite and positive");
  }
  validate_tensor(inputs.current_step, "current_step", {1});
  validate_tensor(inputs.total_step, "total_step", {1});
  if (!has_valid_mask(inputs.latent_mask)) {
    throw std::invalid_argument("latent_mask must contain a valid position");
  }
  if (!has_valid_mask(inputs.text_mask)) {
    throw std::invalid_argument("text_mask must contain a valid position");
  }
}

void invoke_validated_vector(
    const VectorInputs& inputs,
    const RuntimeMetadata& metadata,
    const std::function<void()>& executor) {
  validate_vector_inputs(inputs, metadata);
  if (!executor) {
    throw std::invalid_argument("vector executor callback must not be empty");
  }
  executor();
}

PortableNormalGenerator::PortableNormalGenerator(uint64_t seed)
    : state_(seed == 0 ? 0x12345678abcdef01ULL : seed) {}

uint64_t PortableNormalGenerator::next() {
  state_ ^= state_ << 13;
  state_ ^= state_ >> 7;
  state_ ^= state_ << 17;
  return state_;
}

double PortableNormalGenerator::uniform() {
  return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
}

float PortableNormalGenerator::normal() {
  double first;
  do {
    first = uniform();
  } while (first < 1e-30);
  const double second = uniform();
  return static_cast<float>(
      std::sqrt(-2.0 * std::log(first)) *
      std::cos(6.283185307179586476925286766559 * second));
}

float adjust_duration_for_speed(float duration, float speed) {
  if (!std::isfinite(duration) || duration <= 0.0f) {
    throw std::invalid_argument("duration must be finite and positive");
  }
  if (!std::isfinite(speed) || speed <= 0.0f) {
    throw std::invalid_argument("speed must be finite and positive");
  }
  const double adjusted =
      static_cast<double>(duration) / static_cast<double>(speed);
  if (!std::isfinite(adjusted) ||
      adjusted > std::numeric_limits<float>::max()) {
    throw std::overflow_error("adjusted duration is unrepresentable");
  }
  return static_cast<float>(adjusted);
}

LatentLayout latent_layout(
    float duration,
    int64_t sample_rate,
    int64_t base_chunk_size,
    int64_t chunk_compress_factor,
    int64_t max_latent_length) {
  if (!std::isfinite(duration) || duration <= 0.0f || sample_rate <= 0 ||
      base_chunk_size <= 0 || chunk_compress_factor <= 0 ||
      max_latent_length <= 0) {
    throw std::invalid_argument("invalid duration or latent layout constants");
  }
  const int64_t samples =
      checked_sample_count(duration, sample_rate, "duration");
  if (base_chunk_size >
      std::numeric_limits<int64_t>::max() / chunk_compress_factor) {
    throw std::overflow_error("latent chunk size is unrepresentable");
  }
  const int64_t chunk_size = base_chunk_size * chunk_compress_factor;
  const int64_t latent_length =
      samples / chunk_size + (samples % chunk_size == 0 ? 0 : 1);
  if (latent_length <= 0) {
    throw std::invalid_argument("duration produces no latent positions");
  }
  if (latent_length > max_latent_length) {
    throw std::invalid_argument(
        "predicted duration exceeds exported latent bound");
  }
  return {samples, latent_length};
}

std::vector<float> trim_waveform(
    const std::vector<float>& waveform,
    float duration,
    int64_t sample_rate) {
  if (!std::isfinite(duration) || duration < 0.0f || sample_rate <= 0) {
    throw std::invalid_argument("invalid waveform trim duration");
  }
  const int64_t sample_count =
      checked_sample_count(duration, sample_rate, "trim", true);
  const size_t count =
      std::min(waveform.size(), static_cast<size_t>(sample_count));
  return std::vector<float>(waveform.begin(), waveform.begin() + count);
}

float accumulate_chunk_durations(
    const std::vector<float>& durations,
    float inter_chunk_silence) {
  if (durations.empty()) {
    throw std::invalid_argument("expected at least one chunk duration");
  }
  if (!std::isfinite(inter_chunk_silence) || inter_chunk_silence < 0.0f) {
    throw std::invalid_argument(
        "inter-chunk silence must be finite and nonnegative");
  }
  if (!std::isfinite(durations.front()) || durations.front() <= 0.0f) {
    throw std::invalid_argument("chunk durations must be finite and positive");
  }
  float result = durations.front();
  for (size_t index = 1; index < durations.size(); ++index) {
    if (!std::isfinite(durations[index]) || durations[index] <= 0.0f) {
      throw std::invalid_argument(
          "chunk durations must be finite and positive");
    }
    const float increment = durations[index] + inter_chunk_silence;
    result += increment;
    if (!std::isfinite(increment) || !std::isfinite(result)) {
      throw std::overflow_error("combined duration is unrepresentable");
    }
  }
  return result;
}

std::vector<float> combine_vocoder_chunks(
    const std::vector<std::vector<float>>& waveforms,
    const std::vector<float>& durations,
    int64_t sample_rate,
    float inter_chunk_silence) {
  if (waveforms.empty() || waveforms.size() != durations.size()) {
    throw std::invalid_argument(
        "waveforms and durations must have matching nonzero cardinality");
  }
  const int64_t silence_count =
      checked_sample_count(inter_chunk_silence, sample_rate, "silence", true);
  const float target_duration =
      accumulate_chunk_durations(durations, inter_chunk_silence);
  size_t combined_size = 0;
  for (size_t index = 0; index < waveforms.size(); ++index) {
    combined_size =
        checked_add_size(combined_size, waveforms[index].size(), "combined");
    if (index != 0) {
      combined_size = checked_add_size(
          combined_size, static_cast<size_t>(silence_count), "combined");
    }
  }
  std::vector<float> combined;
  combined.reserve(combined_size);
  for (size_t index = 0; index < waveforms.size(); ++index) {
    if (index != 0) {
      combined.insert(combined.end(), static_cast<size_t>(silence_count), 0.0f);
    }
    combined.insert(
        combined.end(), waveforms[index].begin(), waveforms[index].end());
  }
  return trim_waveform(combined, target_duration, sample_rate);
}

#ifndef SUPERTONIC_PURE_HELPERS_ONLY
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::BackendOptions;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::LoadBackendOptionsMap;

std::vector<c10::Half> to_half(const std::vector<float>& values) {
  std::vector<c10::Half> result;
  result.reserve(values.size());
  for (float value : values) {
    if (!std::isfinite(value) || std::abs(value) > 65504.0f) {
      throw std::invalid_argument(
          "tensor values must be within the finite FP16 range");
    }
    result.emplace_back(value);
  }
  return result;
}

std::vector<float> copy_half_tensor(
    const Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    const char* method) {
  if (tensor.scalar_type() != ScalarType::Half ||
      tensor.sizes().size() != expected_shape.size()) {
    throw std::runtime_error(
        std::string(method) + " returned an incompatible tensor");
  }
  for (size_t index = 0; index < expected_shape.size(); ++index) {
    if (tensor.size(index) != expected_shape[index]) {
      throw std::runtime_error(
          std::string(method) + " returned an incompatible shape");
    }
  }
  ::mlx::core::synchronize();
  const auto* source = tensor.const_data_ptr<c10::Half>();
  std::vector<float> result(tensor.numel());
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<float>(source[index]);
    if (!std::isfinite(result[index])) {
      throw std::runtime_error(
          std::string(method) + " returned a nonfinite FP16 value");
    }
  }
  return result;
}

MetadataValue read_metadata_value(Module& module, const char* name) {
  auto result = module.get(name);
  if (!result.ok()) {
    throw std::runtime_error(std::string("missing metadata value: ") + name);
  }
  if (result->isInt()) {
    return MetadataValue::integer(result->toInt());
  }
  if (result->isBool()) {
    return MetadataValue::boolean(result->toBool());
  }
  if (result->isString()) {
    return MetadataValue::string(std::string(result->toString()));
  }
  throw std::runtime_error(
      std::string("metadata value has unsupported EValue type: ") + name);
}

TensorView domain_view(
    std::vector<int64_t> shape,
    const std::vector<float>& values) {
  return {std::move(shape), &values, true, TensorDtype::Float16};
}

} // namespace

class SupertonicRunner::Impl {
 public:
  Impl(const std::string& pte_path, const std::string& unicode_indexer_path)
      : processor_(unicode_indexer_path),
        module_(std::make_unique<Module>(
            pte_path,
            Module::LoadMode::MmapUseMlockIgnoreErrors)) {
    if (module_->load() != Error::Ok) {
      throw std::runtime_error("failed to load Supertonic PTE: " + pte_path);
    }
    auto methods = module_->method_names();
    if (!methods.ok()) {
      throw std::runtime_error("failed to enumerate Supertonic PTE methods");
    }
    std::set<std::string> method_names;
    for (const auto& method : *methods) {
      method_names.insert(std::string(method));
    }
    std::map<std::string, MetadataValue> metadata_values;
    for (const char* name :
         {"get_sample_rate",
          "get_base_chunk_size",
          "get_chunk_compress_factor",
          "get_flow_steps",
          "get_text_vocabulary_size",
          "get_latent_dim",
          "get_latent_channels",
          "get_max_text_length",
          "get_max_latent_length",
          "get_batch_size",
          "get_activation_dtype",
          "enable_dynamic_shape"}) {
      metadata_values.emplace(name, read_metadata_value(*module_, name));
    }
    metadata_ = validate_metadata_contract(method_names, metadata_values);
    processor_.configure_vocabulary(metadata_.text_vocabulary_size);

    BackendOptions<1> options;
    if (options.set_option(
            ::executorch::backends::mlx::kClearCacheIntervalKey, 1) !=
            Error::Ok ||
        load_options_.set_options(
            ::executorch::backends::mlx::kMLXBackendId, options.view()) !=
            Error::Ok) {
      throw std::runtime_error("failed to configure MLX backend options");
    }
    for (const char* method :
         {"duration_predictor",
          "text_encoder",
          "vector_estimator",
          "vocoder"}) {
      if (module_->load_method(method, nullptr, nullptr, &load_options_) !=
          Error::Ok) {
        throw std::runtime_error(
            std::string("failed to load Supertonic method: ") + method);
      }
    }
  }

  const RuntimeMetadata& metadata() const {
    return metadata_;
  }

  SynthesisResult synthesize(const SynthesisOptions& options) {
    if (options.text.empty()) {
      throw std::invalid_argument("text must not be empty");
    }
    validate_language(options.language);
    if (!std::isfinite(options.speed) || options.speed <= 0.0f) {
      throw std::invalid_argument("speed must be finite and positive");
    }
    if (!std::isfinite(options.inter_chunk_silence) ||
        options.inter_chunk_silence < 0.0f) {
      throw std::invalid_argument(
          "inter-chunk silence must be finite and nonnegative");
    }
    const VoiceStyle style = load_voice_style(
        require_single_voice_style_path(options.voice_style_paths));
    const auto chunks = chunk_text_for_language(options.text, options.language);
    if (chunks.empty()) {
      throw std::invalid_argument("text produced no synthesis chunks");
    }
    const auto synthesis_started = std::chrono::steady_clock::now();
    PortableNormalGenerator generator(options.seed);
    SynthesisResult result;
    std::vector<std::vector<float>> waveforms;
    std::vector<float> durations;
    waveforms.reserve(chunks.size());
    durations.reserve(chunks.size());
    for (const auto& text : chunks) {
      auto chunk = synthesize_chunk(
          text, options.language, style, options.speed, generator);
      waveforms.push_back(std::move(chunk.waveform));
      durations.push_back(chunk.duration_seconds);
    }
    result.waveform = combine_vocoder_chunks(
        waveforms,
        durations,
        metadata_.sample_rate,
        options.inter_chunk_silence);
    result.duration_seconds = static_cast<float>(
        static_cast<double>(result.waveform.size()) / metadata_.sample_rate);
    ::mlx::core::synchronize();
    result.elapsed_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - synthesis_started)
            .count();
    if (result.waveform.empty() ||
        !std::all_of(
            result.waveform.begin(), result.waveform.end(), [](float value) {
              return std::isfinite(value);
            })) {
      throw std::runtime_error(
          "Supertonic synthesis produced an empty or nonfinite waveform");
    }
    return result;
  }

 private:
  struct ChunkResult {
    std::vector<float> waveform;
    float duration_seconds;
  };

  std::vector<EValue> execute(
      const char* method,
      const std::vector<EValue>& inputs) {
    auto outputs = module_->execute(method, inputs);
    if (!outputs.ok()) {
      throw std::runtime_error(std::string("PTE method failed: ") + method);
    }
    ::mlx::core::synchronize();
    return std::move(outputs.get());
  }

  ChunkResult synthesize_chunk(
      const std::string& text,
      const std::string& language,
      const VoiceStyle& style,
      float speed,
      PortableNormalGenerator& generator) {
    const auto size = [](int64_t value) {
      return static_cast<executorch::aten::SizesType>(value);
    };
    TextBatch text_batch = processor_.process({text}, {language});
    const int64_t text_length = text_batch.shape[1];
    if (text_length > metadata_.max_text_length) {
      throw std::invalid_argument(
          "preprocessed text exceeds exported text bound");
    }
    auto text_mask_half = to_half(text_batch.mask);
    auto style_dp_half = to_half(style.dp);
    auto ids_tensor = from_blob(
        text_batch.ids.data(), {1, size(text_length)}, ScalarType::Long);
    auto dp_tensor =
        from_blob(style_dp_half.data(), {1, 8, 16}, ScalarType::Half);
    auto text_mask_tensor = from_blob(
        text_mask_half.data(), {1, 1, size(text_length)}, ScalarType::Half);
    auto duration_outputs = execute(
        "duration_predictor",
        {EValue(ids_tensor), EValue(dp_tensor), EValue(text_mask_tensor)});
    if (duration_outputs.size() != 1 || !duration_outputs[0].isTensor()) {
      throw std::runtime_error("duration_predictor must return one tensor");
    }
    const auto duration_values = copy_half_tensor(
        duration_outputs[0].toTensor(), {1}, "duration_predictor");
    const float duration =
        adjust_duration_for_speed(duration_values.front(), speed);
    const LatentLayout layout = latent_layout(
        duration,
        metadata_.sample_rate,
        metadata_.base_chunk_size,
        metadata_.chunk_compress_factor,
        metadata_.max_latent_length);

    auto style_ttl_half = to_half(style.ttl);
    auto ttl_tensor =
        from_blob(style_ttl_half.data(), {1, 50, 256}, ScalarType::Half);
    auto encoder_outputs = execute(
        "text_encoder",
        {EValue(ids_tensor), EValue(ttl_tensor), EValue(text_mask_tensor)});
    if (encoder_outputs.size() != 1 || !encoder_outputs[0].isTensor()) {
      throw std::runtime_error("text_encoder must return one tensor");
    }
    std::vector<float> text_embedding = copy_half_tensor(
        encoder_outputs[0].toTensor(), {1, 256, text_length}, "text_encoder");

    const size_t latent_channels =
        static_cast<size_t>(metadata_.latent_channels);
    const size_t latent_length = static_cast<size_t>(layout.latent_length);
    if (latent_channels > std::numeric_limits<size_t>::max() / latent_length) {
      throw std::overflow_error("latent tensor size is unrepresentable");
    }
    std::vector<float> latent(latent_channels * latent_length);
    for (float& value : latent) {
      value = generator.normal();
    }
    std::vector<float> latent_mask(layout.latent_length, 1.0f);
    const int64_t valid_latents =
        (layout.waveform_samples +
         metadata_.base_chunk_size * metadata_.chunk_compress_factor - 1) /
        (metadata_.base_chunk_size * metadata_.chunk_compress_factor);
    for (int64_t position = valid_latents; position < layout.latent_length;
         ++position) {
      latent_mask[position] = 0.0f;
      for (int64_t channel = 0; channel < metadata_.latent_channels;
           ++channel) {
        latent[channel * layout.latent_length + position] = 0.0f;
      }
    }
    std::vector<float> current_step{0.0f};
    std::vector<float> total_step{static_cast<float>(metadata_.flow_steps)};

    for (int64_t step = 0; step < metadata_.flow_steps; ++step) {
      current_step[0] = static_cast<float>(step);
      VectorInputs domain{
          domain_view(
              {1, metadata_.latent_channels, layout.latent_length}, latent),
          domain_view({1, 256, text_length}, text_embedding),
          domain_view({1, 50, 256}, style.ttl),
          domain_view({1, 1, layout.latent_length}, latent_mask),
          domain_view({1, 1, text_length}, text_batch.mask),
          domain_view({1}, current_step),
          domain_view({1}, total_step)};
      auto latent_half = to_half(latent);
      auto embedding_half = to_half(text_embedding);
      auto latent_mask_half = to_half(latent_mask);
      auto current_half = to_half(current_step);
      auto total_half = to_half(total_step);
      auto latent_tensor = from_blob(
          latent_half.data(),
          {1, size(metadata_.latent_channels), size(layout.latent_length)},
          ScalarType::Half);
      auto embedding_tensor = from_blob(
          embedding_half.data(), {1, 256, size(text_length)}, ScalarType::Half);
      auto latent_mask_tensor = from_blob(
          latent_mask_half.data(),
          {1, 1, size(layout.latent_length)},
          ScalarType::Half);
      auto current_tensor =
          from_blob(current_half.data(), {1}, ScalarType::Half);
      auto total_tensor = from_blob(total_half.data(), {1}, ScalarType::Half);
      std::vector<EValue> vector_outputs;
      invoke_validated_vector(domain, metadata_, [&] {
        vector_outputs = execute(
            "vector_estimator",
            {EValue(latent_tensor),
             EValue(embedding_tensor),
             EValue(ttl_tensor),
             EValue(latent_mask_tensor),
             EValue(text_mask_tensor),
             EValue(current_tensor),
             EValue(total_tensor)});
      });
      if (vector_outputs.size() != 1 || !vector_outputs[0].isTensor()) {
        throw std::runtime_error("vector_estimator must return one tensor");
      }
      latent = copy_half_tensor(
          vector_outputs[0].toTensor(),
          {1, metadata_.latent_channels, layout.latent_length},
          "vector_estimator");
    }

    auto latent_half = to_half(latent);
    auto latent_tensor = from_blob(
        latent_half.data(),
        {1, size(metadata_.latent_channels), size(layout.latent_length)},
        ScalarType::Half);
    auto vocoder_outputs = execute("vocoder", {EValue(latent_tensor)});
    if (vocoder_outputs.size() != 1 || !vocoder_outputs[0].isTensor()) {
      throw std::runtime_error("vocoder must return one tensor");
    }
    const int64_t chunk_size =
        metadata_.base_chunk_size * metadata_.chunk_compress_factor;
    if (layout.latent_length >
        std::numeric_limits<int64_t>::max() / chunk_size) {
      throw std::overflow_error("vocoder sample count is unrepresentable");
    }
    const int64_t produced_samples = layout.latent_length * chunk_size;
    auto waveform = copy_half_tensor(
        vocoder_outputs[0].toTensor(), {1, produced_samples}, "vocoder");
    return {std::move(waveform), duration};
  }

  UnicodeProcessor processor_;
  std::unique_ptr<Module> module_;
  LoadBackendOptionsMap load_options_;
  RuntimeMetadata metadata_;
};

SupertonicRunner::SupertonicRunner(
    const std::string& pte_path,
    const std::string& unicode_indexer_path)
    : impl_(std::make_unique<Impl>(pte_path, unicode_indexer_path)) {}

SupertonicRunner::~SupertonicRunner() = default;

const RuntimeMetadata& SupertonicRunner::metadata() const {
  return impl_->metadata();
}

SynthesisResult SupertonicRunner::synthesize(const SynthesisOptions& options) {
  return impl_->synthesize(options);
}

#endif

} // namespace supertonic
