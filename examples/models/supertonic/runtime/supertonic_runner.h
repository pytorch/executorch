/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "style_loader.h"
#include "text_processor.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace supertonic {

struct RuntimeMetadata {
  int64_t sample_rate = 44100;
  int64_t base_chunk_size = 512;
  int64_t chunk_compress_factor = 6;
  int64_t flow_steps = 5;
  int64_t text_vocabulary_size = 0;
  int64_t latent_dim = 24;
  int64_t latent_channels = 144;
  int64_t max_text_length = 512;
  int64_t max_latent_length = 512;
  int64_t batch_size = 1;
  std::string activation_dtype = "float16";
  bool dynamic_shapes = true;
};

enum class MetadataValueType {
  Integer,
  Boolean,
  String,
};

struct MetadataValue {
  MetadataValueType type;
  int64_t integer_value = 0;
  bool boolean_value = false;
  std::string string_value;

  static MetadataValue integer(int64_t value);
  static MetadataValue boolean(bool value);
  static MetadataValue string(std::string value);
};

RuntimeMetadata validate_metadata_contract(
    const std::set<std::string>& method_names,
    const std::map<std::string, MetadataValue>& metadata_values);

enum class TensorDtype {
  Float16,
  Other,
};

struct TensorView {
  std::vector<int64_t> shape;
  const std::vector<float>* values;
  bool contiguous;
  TensorDtype dtype;
};

struct VectorInputs {
  TensorView noisy_latent;
  TensorView text_emb;
  TensorView style_ttl;
  TensorView latent_mask;
  TensorView text_mask;
  TensorView current_step;
  TensorView total_step;
};

void validate_vector_inputs(
    const VectorInputs& inputs,
    const RuntimeMetadata& metadata);
void invoke_validated_vector(
    const VectorInputs& inputs,
    const RuntimeMetadata& metadata,
    const std::function<void()>& executor);

class PortableNormalGenerator {
 public:
  explicit PortableNormalGenerator(uint64_t seed);
  float normal();

 private:
  uint64_t next();
  double uniform();
  uint64_t state_;
};

struct LatentLayout {
  int64_t waveform_samples;
  int64_t latent_length;
};

float adjust_duration_for_speed(float duration, float speed);
LatentLayout latent_layout(
    float duration,
    int64_t sample_rate,
    int64_t base_chunk_size,
    int64_t chunk_compress_factor,
    int64_t max_latent_length);
std::vector<float> trim_waveform(
    const std::vector<float>& waveform,
    float duration,
    int64_t sample_rate);
float accumulate_chunk_durations(
    const std::vector<float>& durations,
    float inter_chunk_silence);
std::vector<float> combine_vocoder_chunks(
    const std::vector<std::vector<float>>& waveforms,
    const std::vector<float>& durations,
    int64_t sample_rate,
    float inter_chunk_silence);

struct SynthesisOptions {
  std::string text;
  std::string language = "en";
  std::vector<std::string> voice_style_paths;
  float speed = 1.05f;
  uint64_t seed = 42;
  float inter_chunk_silence = 0.3f;
};

struct SynthesisResult {
  std::vector<float> waveform;
  float duration_seconds = 0.0f;
  double elapsed_seconds = 0.0;
};

class SupertonicRunner {
 public:
  SupertonicRunner(
      const std::string& pte_path,
      const std::string& unicode_indexer_path);
  ~SupertonicRunner();

  SupertonicRunner(const SupertonicRunner&) = delete;
  SupertonicRunner& operator=(const SupertonicRunner&) = delete;

  const RuntimeMetadata& metadata() const;
  SynthesisResult synthesize(const SynthesisOptions& options);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace supertonic
