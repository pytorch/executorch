/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/sentencepiece.h>

namespace phonon {

// Collapses whitespace, capitalizes the first letter and ends the text with a
// period if it ends with a letter or digit. Returns the text and the number of
// frames to generate after the end-of-speech logit fires (3 for texts of up to
// 4 words, else 1). Same rules as run_phonon.py.
std::pair<std::string, int> prepare_text(const std::string& text);

// Packs whole sentences into chunks of at most max_tokens tokens; a longer
// sentence is split at word boundaries. Same rules as run_phonon.py.
std::vector<std::string> split_sentences(
    const std::string& text,
    const std::function<size_t(const std::string&)>& count_tokens,
    size_t max_tokens);

// Gaussian noise for the flow head, identical to NoiseGenerator in
// run_phonon.py: splitmix64 uniforms and the Box-Muller transform.
class NoiseGenerator {
 public:
  explicit NoiseGenerator(uint64_t seed) : state_(seed) {}
  void fill(float* out, size_t count, double stddev);

 private:
  double uniform();
  uint64_t state_;
};

// Speaker latents, time-major [frames, dim].
struct Voice {
  std::vector<float> latents;
  int64_t frames = 0;
};

struct Chunk {
  std::string text;
  std::vector<uint64_t> tokens;
  int64_t frames = 0;
};

struct Synthesis {
  std::vector<float> pcm;
  std::vector<Chunk> chunks;
  int64_t frames = 0;
};

class Runner {
 public:
  // data_path is the external data file of the program, if any
  // (aoti_cuda_blob.ptd for CUDA).
  Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& data_path = "");

  // Reads a Phonon voice .safetensors file (speaker_wavs [1, dim, frames]).
  Voice load_voice(const std::string& path) const;

  // temperature < 0 uses the model's default. on_audio, if set, receives the
  // PCM of every frame as soon as it is decoded.
  Synthesis synthesize(
      const std::string& text,
      const Voice& voice,
      double temperature = -1,
      uint64_t seed = 0,
      const std::function<void(const float*, size_t)>& on_audio = nullptr);

  std::vector<uint64_t> tokenize(const std::string& text) const;

  int64_t sample_rate() const {
    return sample_rate_;
  }

 private:
  struct FlowOutput {
    std::vector<float> latent;
    float eos_logit;
    std::vector<float> next_embeds;
  };
  FlowOutput flow(const std::vector<float>& embeds, int64_t pos);
  FlowOutput flow(const float* embeds, int64_t seq, int64_t pos);

  executorch::extension::Module model_;
  tokenizers::SPTokenizer tokenizer_;
  int64_t sample_rate_, frame_samples_, dim_, latent_dim_, max_context_,
      max_prefill_;
  double frame_rate_, eos_threshold_, temperature_;
  std::vector<float> bos_embeds_, noise_;
  // Shapes of the Mimi state passed in and out of mimi_decode; empty when the
  // program keeps the state (mimi_state_in_program) and takes a reset flag.
  std::vector<std::vector<executorch::aten::SizesType>> mimi_state_sizes_;
  bool mimi_state_in_program_ = false;
};

} // namespace phonon
