/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_runner.h"

#include "reverb_audio.h"
#include "reverb_text.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace reverb {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::from_blob;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::EValue;

constexpr float kLongestSeconds = 30.0f; // texts estimated longer are split
constexpr float kPieceSeconds = 15.0f; // into pieces of about this length
constexpr float kEndCutSeconds = 0.2f; // the codec can end on a noise burst
constexpr float kLongReferenceSeconds = 15.0f;

class Stopwatch {
 public:
  explicit Stopwatch(double& total)
      : total_(total), start_(std::chrono::steady_clock::now()) {}
  ~Stopwatch() {
    total_ +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count();
  }

 private:
  double& total_;
  std::chrono::steady_clock::time_point start_;
};

int64_t read_int(Module& module, const char* name) {
  auto value = module.get(name);
  if (!value.ok() || !value->isInt()) {
    throw std::runtime_error(std::string("program lacks metadata ") + name);
  }
  return value->toInt();
}

std::vector<float> to_floats(const Tensor& tensor) {
  std::vector<float> out(tensor.numel());
  if (tensor.scalar_type() == ScalarType::Float) {
    std::memcpy(
        out.data(), tensor.const_data_ptr<float>(), out.size() * sizeof(float));
  } else if (tensor.scalar_type() == ScalarType::Half) {
    const auto* data = tensor.const_data_ptr<::executorch::aten::Half>();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = static_cast<float>(data[i]);
    }
  } else {
    throw std::runtime_error("expected a float or half output");
  }
  return out;
}

std::vector<int64_t> to_ints(const Tensor& tensor) {
  std::vector<int64_t> out(tensor.numel());
  if (tensor.scalar_type() == ScalarType::Long) {
    std::memcpy(
        out.data(),
        tensor.const_data_ptr<int64_t>(),
        out.size() * sizeof(int64_t));
  } else if (tensor.scalar_type() == ScalarType::Int) {
    const auto* data = tensor.const_data_ptr<int32_t>();
    std::copy(data, data + out.size(), out.begin());
  } else {
    throw std::runtime_error("expected an int32 or int64 output");
  }
  return out;
}

std::vector<EValue> run(
    Module& module,
    const char* method,
    const std::vector<EValue>& inputs,
    size_t expected_outputs) {
  auto outputs = module.execute(method, inputs);
  if (!outputs.ok()) {
    throw std::runtime_error(
        std::string("method ") + method + " failed with error " +
        std::to_string(static_cast<int>(outputs.error())));
  }
  if (outputs->size() != expected_outputs) {
    throw std::runtime_error(
        std::string("method ") + method +
        " returned an unexpected number of outputs");
  }
  return std::move(*outputs);
}

} // namespace

class ReverbRunner::Impl {
 public:
  Impl(
      const std::string& backbone_path,
      const std::string& codec_path,
      const std::string& tokenizer_path)
      : backbone_(backbone_path, Module::LoadMode::MmapUseMlockIgnoreErrors),
        codec_(codec_path, Module::LoadMode::MmapUseMlockIgnoreErrors) {
    if (tokenizer_.load(tokenizer_path) != tokenizers::Error::Ok) {
      throw std::runtime_error("cannot load tokenizer " + tokenizer_path);
    }
    limits_.prompt_tokens = read_int(backbone_, "get_max_prompt_tokens");
    limits_.reference_frames = read_int(backbone_, "get_max_reference_frames");
    limits_.target_frames = read_int(backbone_, "get_max_target_frames");
    limits_.sample_rate = read_int(codec_, "get_sample_rate");
    limits_.hop_length = read_int(codec_, "get_hop_length");
    limits_.min_encode_frames = read_int(codec_, "get_min_encode_frames");
    limits_.max_encode_frames = read_int(codec_, "get_max_encode_frames");
    if (read_int(backbone_, "get_num_codebooks") != kCodebooks ||
        read_int(backbone_, "get_mask_code") != kMask ||
        read_int(codec_, "get_num_codebooks") != kCodebooks ||
        limits_.sample_rate != limits_.hop_length * kFrameRate) {
      throw std::runtime_error("the programs do not match this runner");
    }
  }

  void load(bool with_encoder) {
    for (const char* method : {"prefill", "step", "polish"}) {
      check_load(backbone_, method);
    }
    check_load(codec_, "decode");
    if (with_encoder) {
      check_load(codec_, "encode");
    }
  }

  Voice clone_voice(
      const std::string& wav_path,
      const std::string& transcript,
      const std::string& language) {
    if (!is_supported_language(language)) {
      throw std::invalid_argument("unsupported language " + language);
    }
    Audio audio = read_wav(wav_path);
    std::vector<float> wav = resample(
        audio.samples,
        audio.sample_rate,
        static_cast<int>(limits_.sample_rate));
    raise_quiet(wav);
    // Whole codec frames, as the training references were.
    const int64_t hop = limits_.hop_length;
    wav.resize((wav.size() + hop - 1) / hop * hop, 0.0f);
    const int64_t frames = static_cast<int64_t>(wav.size()) / hop;
    if (frames < limits_.min_encode_frames ||
        frames > limits_.max_encode_frames) {
      throw std::invalid_argument(
          "the reference must last between " +
          std::to_string(limits_.min_encode_frames) + " and " +
          std::to_string(limits_.max_encode_frames) +
          " codec frames (25 per second)");
    }
    if (frames > kLongReferenceSeconds * kFrameRate) {
      ET_LOG(Info, "References over 15 s tend to make worse takes.");
    }
    auto input = from_blob(
        wav.data(), {static_cast<int>(wav.size())}, ScalarType::Float);
    auto outputs = run(codec_, "encode", {EValue(input)}, 1);
    Voice voice;
    voice.codes = to_ints(outputs[0].toTensor());
    if (voice.codes.size() != static_cast<size_t>(kCodebooks * frames)) {
      throw std::runtime_error("encode returned an unexpected shape");
    }
    voice.transcript = transcript;
    voice.language = language;
    voice.speech_frames = static_cast<int>(std::lround(
        speech_length(wav, static_cast<int>(limits_.sample_rate)) *
        kFrameRate));
    return voice;
  }

  std::vector<float> generate(
      const GenerateOptions& options,
      const Voice& voice) {
    validate(options.sampler);
    if (!is_supported_language(options.language)) {
      throw std::invalid_argument("unsupported language " + options.language);
    }
    if (!(options.speed > 0.0f)) {
      throw std::invalid_argument("speed must be positive");
    }
    if (voice.frames() > limits_.reference_frames) {
      throw std::invalid_argument(
          "the voice has " + std::to_string(voice.frames()) +
          " frames; the program takes at most " +
          std::to_string(limits_.reference_frames));
    }
    const int pace = voice.pace_frames(options.pace_whole_recording);
    std::vector<std::pair<std::string, int>> pieces;
    if (options.duration > 0.0f) {
      pieces.emplace_back(
          options.text,
          std::max(
              1, static_cast<int>(std::lround(options.duration * kFrameRate))));
    } else {
      for (const auto& part : split_text(
               options.text,
               voice.transcript,
               pace,
               static_cast<int>(kLongestSeconds * kFrameRate),
               static_cast<int>(kPieceSeconds * kFrameRate))) {
        const double estimate =
            estimate_frames(part, voice.transcript, pace) / options.speed;
        pieces.emplace_back(
            part, std::max(1, static_cast<int>(std::nearbyint(estimate))));
      }
    }

    timings_ = Timings();
    std::vector<float> audio;
    for (size_t i = 0; i < pieces.size(); ++i) {
      const auto& [text, frames] = pieces[i];
      std::vector<int64_t> codes =
          sample(text, frames, options, voice, options.seed + i);
      std::vector<float> take = decode(codes, frames);
      const size_t cut =
          static_cast<size_t>(kEndCutSeconds * limits_.sample_rate);
      take.resize(take.size() > cut ? take.size() - cut : 0);
      if (options.trim) {
        take = trim_silence(take, static_cast<int>(limits_.sample_rate));
      }
      audio.insert(audio.end(), take.begin(), take.end());
    }
    return audio;
  }

  const Limits& limits() const {
    return limits_;
  }

  const Timings& timings() const {
    return timings_;
  }

 private:
  static void check_load(Module& module, const char* method) {
    if (module.load_method(method) != ::executorch::runtime::Error::Ok) {
      throw std::runtime_error(std::string("cannot load method ") + method);
    }
  }

  // Fills a fully masked [8, frames] grid of codes for one piece of text.
  std::vector<int64_t> sample(
      const std::string& text,
      int frames,
      const GenerateOptions& options,
      const Voice& voice,
      uint64_t seed) {
    if (frames > limits_.target_frames) {
      throw std::invalid_argument(
          "a piece needs " + std::to_string(frames) +
          " frames; the program takes at most " +
          std::to_string(limits_.target_frames) +
          " (split long sentences, or lower --duration)");
    }
    auto encoded = tokenizer_.encode(
        build_prompt(text, voice.transcript, options.language), 0, 0);
    if (!encoded.ok()) {
      throw std::runtime_error("cannot tokenize the prompt");
    }
    std::vector<int64_t> ids(encoded->begin(), encoded->end());
    if (static_cast<int64_t>(ids.size()) > limits_.prompt_tokens) {
      throw std::invalid_argument(
          "the prompt has " + std::to_string(ids.size()) +
          " tokens; the program takes at most " +
          std::to_string(limits_.prompt_tokens));
    }

    // The prompt and the reference run once; every pass reads their cache.
    std::vector<int64_t> reference = voice.codes;
    auto ids_tensor =
        from_blob(ids.data(), {static_cast<int>(ids.size())}, ScalarType::Long);
    auto reference_tensor = from_blob(
        reference.data(), {kCodebooks, voice.frames()}, ScalarType::Long);
    std::vector<uint8_t> cache_bytes;
    TensorPtr cache;
    {
      Stopwatch stopwatch(timings_.prefill);
      const Tensor prefilled =
          run(backbone_,
              "prefill",
              {EValue(ids_tensor), EValue(reference_tensor)},
              1)[0]
              .toTensor();
      cache_bytes.assign(
          prefilled.const_data_ptr<uint8_t>(),
          prefilled.const_data_ptr<uint8_t>() + prefilled.nbytes());
      std::vector<::executorch::aten::SizesType> sizes(
          prefilled.sizes().begin(), prefilled.sizes().end());
      cache =
          make_tensor_ptr(sizes, cache_bytes.data(), prefilled.scalar_type());
    }

    std::vector<int64_t> tokens(
        static_cast<size_t>(kCodebooks) * frames, kMask);
    auto tokens_tensor =
        from_blob(tokens.data(), {kCodebooks, frames}, ScalarType::Long);
    const std::vector<float> schedule = commit_schedule(options.sampler);
    Noise noise(seed);
    for (int step = 0; step < options.sampler.steps; ++step) {
      Stopwatch stopwatch(timings_.steps);
      auto outputs =
          run(backbone_, "step", {EValue(tokens_tensor), EValue(cache)}, 3);
      const std::vector<float> confidence = to_floats(outputs[0].toTensor());
      const std::vector<int64_t> prediction = to_ints(outputs[1].toTensor());
      const std::vector<float> held = to_floats(outputs[2].toTensor());
      const size_t slots = static_cast<size_t>(kMainCodebooks) * frames;
      if (confidence.size() != slots || prediction.size() != slots ||
          held.size() != slots) {
        throw std::runtime_error("step returned an unexpected shape");
      }
      apply_main_pass(
          tokens,
          frames,
          {confidence.data(), prediction.data(), held.data()},
          step,
          options.sampler,
          schedule,
          noise);
    }

    int64_t reference_frames = voice.frames();
    auto reference_frames_tensor =
        from_blob(&reference_frames, {1}, ScalarType::Long);
    std::vector<int64_t> fine;
    {
      Stopwatch stopwatch(timings_.polish);
      auto outputs =
          run(backbone_,
              "polish",
              {EValue(tokens_tensor),
               EValue(reference_frames_tensor),
               EValue(cache)},
              1);
      fine = to_ints(outputs[0].toTensor());
    }
    if (fine.size() !=
        static_cast<size_t>(kCodebooks - kMainCodebooks) * frames) {
      throw std::runtime_error("polish returned an unexpected shape");
    }
    std::copy(
        fine.begin(),
        fine.end(),
        tokens.begin() + static_cast<size_t>(kMainCodebooks) * frames);
    return tokens;
  }

  std::vector<float> decode(std::vector<int64_t>& codes, int frames) {
    Stopwatch stopwatch(timings_.decode);
    auto input =
        from_blob(codes.data(), {kCodebooks, frames}, ScalarType::Long);
    auto outputs = run(codec_, "decode", {EValue(input)}, 1);
    std::vector<float> audio = to_floats(outputs[0].toTensor());
    if (audio.size() != static_cast<size_t>(frames * limits_.hop_length)) {
      throw std::runtime_error("decode returned an unexpected shape");
    }
    return audio;
  }

  Module backbone_;
  Module codec_;
  tokenizers::HFTokenizer tokenizer_;
  Limits limits_;
  Timings timings_;
};

ReverbRunner::ReverbRunner(
    const std::string& backbone_path,
    const std::string& codec_path,
    const std::string& tokenizer_path)
    : impl_(std::make_unique<Impl>(backbone_path, codec_path, tokenizer_path)) {
}

ReverbRunner::~ReverbRunner() = default;

const Limits& ReverbRunner::limits() const {
  return impl_->limits();
}

void ReverbRunner::load(bool with_encoder) {
  impl_->load(with_encoder);
}

Voice ReverbRunner::clone_voice(
    const std::string& wav_path,
    const std::string& transcript,
    const std::string& language) {
  return impl_->clone_voice(wav_path, transcript, language);
}

std::vector<float> ReverbRunner::generate(
    const GenerateOptions& options,
    const Voice& voice) {
  return impl_->generate(options, voice);
}

const Timings& ReverbRunner::last_timings() const {
  return impl_->timings();
}

} // namespace reverb
