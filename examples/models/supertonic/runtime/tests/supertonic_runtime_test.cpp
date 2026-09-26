/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../style_loader.h"
#include "../supertonic_runner.h"
#include "../text_processor.h"
#include "../wav_writer.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace supertonic;

void check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void check_throws(Function&& function, const std::string& expected) {
  try {
    function();
  } catch (const std::exception& error) {
    check(
        std::string(error.what()).find(expected) != std::string::npos,
        "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected error containing: " + expected);
}

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
          ("supertonic-runtime-" + std::to_string(random()) + "-" +
           std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
    }
    throw std::runtime_error("failed to create temporary test directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::filesystem::path path(const std::string& name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

TempDirectory& temp_directory() {
  static TempDirectory directory;
  return directory;
}

std::filesystem::path temp_path(const std::string& name) {
  return temp_directory().path(name);
}

void write_ascii_indexer(const std::filesystem::path& path) {
  std::ofstream file(path);
  file << "[";
  for (int index = 0; index < 1024; ++index) {
    if (index != 0) {
      file << ",";
    }
    file << index;
  }
  file << "]";
}

std::string style_json(float ttl_value, float dp_value) {
  std::string result = R"({"style_ttl":{"dims":[1,50,256],"data":[)";
  for (int index = 0; index < 50 * 256; ++index) {
    result += (index == 0 ? "" : ",") + std::to_string(ttl_value);
  }
  result += R"(]},"style_dp":{"dims":[1,8,16],"data":[)";
  for (int index = 0; index < 8 * 16; ++index) {
    result += (index == 0 ? "" : ",") + std::to_string(dp_value);
  }
  return result + "]}}";
}

std::string nested_style_json(float value, bool ragged_dp = false) {
  std::string result = R"({"style_ttl":{"dims":[1,50,256],"data":[[)";
  for (int row = 0; row < 50; ++row) {
    result += row == 0 ? "[" : ",[";
    for (int column = 0; column < 256; ++column) {
      result += (column == 0 ? "" : ",") + std::to_string(value);
    }
    result += "]";
  }
  result += R"(]]},"style_dp":{"dims":[1,8,16],"data":[[)";
  for (int row = 0; row < 8; ++row) {
    result += row == 0 ? "[" : ",[";
    const int columns = ragged_dp && row == 7 ? 15 : 16;
    for (int column = 0; column < columns; ++column) {
      result += (column == 0 ? "" : ",") + std::to_string(value);
    }
    result += "]";
  }
  return result + "]]}}";
}

TensorView view(
    std::vector<int64_t> shape,
    const std::vector<float>& values,
    bool contiguous = true,
    TensorDtype dtype = TensorDtype::Float16) {
  return TensorView{std::move(shape), &values, contiguous, dtype};
}

void test_preprocessing() {
  check(
      preprocess_text("Caf\xC3\xA9", "en") == "<en>Cafe\xCC\x81.</en>",
      "NFKD preprocessing mismatch");
  check(
      preprocess_text(
          "\xE2\x80\x9CHello\xE2\x80\x9D \xE2\x80\x94 world_"
          "\xF0\x9F\x99\x82 @ x \xE2\x99\xA5 e.g., i.e., [done]",
          "en") ==
          "<en>\"Hello\" - world at x for example, that is, done.</en>",
      "cleanup preprocessing mismatch");
  check_throws(
      [] { (void)preprocess_text("hello", "xx"); }, "Invalid language: xx");
  check(
      preprocess_text(
          "one\xC2\xA0"
          "two\xE2\x80\x83"
          "three\xE2\x80\xA8"
          "four",
          "en") == "<en>one two three four.</en>",
      "Unicode whitespace normalization mismatch");
  check(
      preprocess_text("word\t,", "en") == "<en>word ,</en>",
      "tab-before-punctuation ordering mismatch");
  check(
      preprocess_text(
          "word\xE2\x80\xA8"
          ",",
          "en") == "<en>word ,</en>",
      "Unicode-whitespace-before-punctuation ordering mismatch");
  check(
      preprocess_text(
          "one\ttwo\xE2\x80\xA8"
          "three",
          "en") == "<en>one two three.</en>",
      "between-word Unicode whitespace mismatch");

  const auto indexer = temp_path("indexer.json");
  write_ascii_indexer(indexer);
  UnicodeProcessor processor(indexer.string());
  processor.configure_vocabulary(1024);
  auto batch = processor.process({"A", "Hi!"}, {"en", "en"});
  check(batch.shape == std::vector<int64_t>({2, 12}), "text shape mismatch");
  check(
      batch.ids[0] == 60 && batch.ids[10] == 62 && batch.ids[11] == 0,
      "text ids mismatch");
  check(batch.mask[10] == 1.0f && batch.mask[11] == 0.0f, "text mask mismatch");
  check_throws([&] { (void)processor.process({}, {}); }, "at least one text");
  check_throws(
      [&] { (void)processor.process({"a", "b"}, {"en"}); }, "same cardinality");
  const auto unsupported_indexer = temp_path("unsupported-indexer.json");
  write_ascii_indexer(unsupported_indexer);
  {
    std::ifstream input(unsupported_indexer);
    nlohmann::json values = nlohmann::json::parse(input);
    values[65] = -1;
    std::ofstream(unsupported_indexer) << values;
  }
  UnicodeProcessor unsupported(unsupported_indexer.string());
  unsupported.configure_vocabulary(1024);
  check_throws(
      [&] { (void)unsupported.process({"A"}, {"en"}); },
      "unsupported Unicode codepoint 65");
  check_throws(
      [&] { processor.configure_vocabulary(100); },
      "outside the text vocabulary");

  const auto invalid_type_indexer = temp_path("invalid-type-indexer.json");
  write_ascii_indexer(invalid_type_indexer);
  {
    std::ifstream input(invalid_type_indexer);
    nlohmann::json values = nlohmann::json::parse(input);
    values[65] = true;
    std::ofstream(invalid_type_indexer) << values;
  }
  check_throws(
      [&] { UnicodeProcessor invalid(invalid_type_indexer.string()); },
      "tokens must be integers");

  std::filesystem::remove(indexer);
  std::filesystem::remove(unsupported_indexer);
  std::filesystem::remove(invalid_type_indexer);
}

void test_chunking() {
  const std::string first(60, 'x');
  const std::string second(60, 'y');
  const std::string third(10, 'z');
  const std::string text = first + ". " + second + ". " + third + ".";
  const auto korean = chunk_text_for_language(text, "ko");
  check(korean.size() == 2, "Korean threshold did not split");
  check(korean[0] == first + ".", "first Korean chunk mismatch");
  check(
      korean[1] == second + ". " + third + ".", "second Korean chunk mismatch");
  check(
      chunk_text_for_language(text, "en") == std::vector<std::string>({text}),
      "English threshold split unexpectedly");
  const std::string oversized(301, 'q');
  check(
      chunk_text_for_language(oversized + ".", "en") ==
          std::vector<std::string>({oversized + "."}),
      "soft limit split one sentence");
  check(
      chunk_text("Dr. Smith left. Next sentence.", 20) ==
          std::vector<std::string>({"Dr. Smith left.", "Next sentence."}),
      "abbreviation sentence split mismatch");
  const auto repeat = [](const std::string& value, size_t count) {
    std::string result;
    for (size_t index = 0; index < count; ++index) {
      result += value;
    }
    return result;
  };
  const std::string ga = repeat("\xEA\xB0\x80", 60);
  const std::string na = repeat("\xEB\x82\x98", 60);
  const std::string da = repeat("\xEB\x8B\xA4", 10);
  check(
      chunk_text_for_language(ga + ". " + na + ". " + da + ".", "ko") ==
          std::vector<std::string>({ga + ".", na + ". " + da + "."}),
      "chunk threshold must count Unicode characters");
  const std::string a = repeat("\xE3\x81\x82", 60) + "\xE3\x80\x82";
  const std::string i = repeat("\xE3\x81\x84", 59) + "\xEF\xBC\x81\xEF\xBC\x9F";
  const std::string u = repeat("\xE3\x81\x86", 10) + "\xEF\xBC\x9F";
  check(
      chunk_text_for_language(a + i + u, "ja") ==
          std::vector<std::string>({a, i + " " + u}),
      "CJK sentence terminators must split without spaces");
}

void test_style_loading(const std::string& published_style_path) {
  const auto nested = temp_path("style-nested.json");
  std::ofstream(nested) << nested_style_json(5.0f);
  const VoiceStyle published = load_voice_style(nested.string());
  check(
      published.ttl.front() == 5.0f && published.dp.back() == 5.0f,
      "published nested style data mismatch");
  check_throws(
      [] { (void)require_single_voice_style_path({}); },
      "exactly one voice style");
  check_throws(
      [] { (void)require_single_voice_style_path({"a.json", "b.json"}); },
      "exactly one voice style");

  const auto flat = temp_path("style-flat.json");
  std::ofstream(flat) << style_json(1.0f, 2.0f);
  check_throws(
      [&] { (void)load_voice_style(flat.string()); },
      "style_ttl.data must have nested shape [1, 50, 256]");

  const auto ragged = temp_path("style-ragged.json");
  std::ofstream(ragged) << nested_style_json(1.0f, true);
  check_throws(
      [&] { (void)load_voice_style(ragged.string()); },
      "style_dp.data must have nested shape [1, 8, 16]");

  const auto overflow = temp_path("style-overflow.json");
  std::ofstream(overflow) << nested_style_json(65505.0f);
  check_throws(
      [&] { (void)load_voice_style(overflow.string()); }, "finite FP16 range");

  if (!published_style_path.empty()) {
    const VoiceStyle actual = load_voice_style(published_style_path);
    check(
        actual.ttl.size() == 50 * 256 && actual.dp.size() == 8 * 16,
        "published voice style tensor sizes mismatch");
    const auto all_finite = [](const std::vector<float>& values) {
      return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
      });
    };
    check(
        all_finite(actual.ttl) && all_finite(actual.dp),
        "published voice style contains nonfinite values");
  }
}

void test_portable_normal_generator() {
  PortableNormalGenerator generator(42);
  const std::vector<float> expected = {
      -4.44757128f,
      0.608476877f,
      -1.46980774f,
      -0.868334234f,
      -0.509428322f,
      0.75734967f,
      -0.881577611f,
      -0.204967409f};
  for (float value : expected) {
    check(
        std::abs(generator.normal() - value) < 1e-6f,
        "seed-42 normal golden mismatch");
  }
  PortableNormalGenerator repeated(42);
  check(
      repeated.normal() == expected.front(), "normal generator not repeatable");
}

std::map<std::string, MetadataValue> valid_metadata_values() {
  return {
      {"get_sample_rate", MetadataValue::integer(44100)},
      {"get_base_chunk_size", MetadataValue::integer(512)},
      {"get_chunk_compress_factor", MetadataValue::integer(6)},
      {"get_flow_steps", MetadataValue::integer(5)},
      {"get_text_vocabulary_size", MetadataValue::integer(8322)},
      {"get_latent_dim", MetadataValue::integer(24)},
      {"get_latent_channels", MetadataValue::integer(144)},
      {"get_max_text_length", MetadataValue::integer(512)},
      {"get_max_latent_length", MetadataValue::integer(512)},
      {"get_batch_size", MetadataValue::integer(1)},
      {"get_activation_dtype", MetadataValue::string("float16")},
      {"enable_dynamic_shape", MetadataValue::boolean(true)},
  };
}

std::set<std::string> valid_method_names() {
  return {
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
      "enable_dynamic_shape",
  };
}

void test_metadata_contract() {
  const RuntimeMetadata metadata =
      validate_metadata_contract(valid_method_names(), valid_metadata_values());
  check(
      metadata.sample_rate == 44100 && metadata.latent_channels == 144 &&
          metadata.text_vocabulary_size == 8322,
      "valid metadata contract mismatch");

  auto missing_methods = valid_method_names();
  missing_methods.erase("vocoder");
  check_throws(
      [&] {
        (void)validate_metadata_contract(
            missing_methods, valid_metadata_values());
      },
      "missing methods: vocoder");

  missing_methods = valid_method_names();
  missing_methods.erase("get_text_vocabulary_size");
  check_throws(
      [&] {
        (void)validate_metadata_contract(
            missing_methods, valid_metadata_values());
      },
      "missing methods: get_text_vocabulary_size");

  auto unexpected_methods = valid_method_names();
  unexpected_methods.insert("forward");
  check_throws(
      [&] {
        (void)validate_metadata_contract(
            unexpected_methods, valid_metadata_values());
      },
      "unexpected methods: forward");

  auto wrong_type = valid_metadata_values();
  wrong_type["get_sample_rate"] = MetadataValue::boolean(true);
  check_throws(
      [&] {
        (void)validate_metadata_contract(valid_method_names(), wrong_type);
      },
      "get_sample_rate must be an integer");

  auto invalid_vocabulary = valid_metadata_values();
  invalid_vocabulary["get_text_vocabulary_size"] = MetadataValue::integer(0);
  check_throws(
      [&] {
        (void)validate_metadata_contract(
            valid_method_names(), invalid_vocabulary);
      },
      "PTE metadata is incompatible");

  auto missing_value = valid_metadata_values();
  missing_value.erase("get_flow_steps");
  check_throws(
      [&] {
        (void)validate_metadata_contract(valid_method_names(), missing_value);
      },
      "missing metadata value: get_flow_steps");
}

void test_vector_domain_validation() {
  RuntimeMetadata metadata;
  metadata.latent_channels = 144;
  metadata.max_text_length = 512;
  metadata.max_latent_length = 512;

  std::vector<float> latent(144 * 3, 0.5f);
  std::vector<float> text_emb(256 * 4, 0.5f);
  std::vector<float> style(50 * 256, 0.5f);
  std::vector<float> latent_mask(3, 1.0f);
  std::vector<float> text_mask(4, 1.0f);
  std::vector<float> current_step{0.0f};
  std::vector<float> total_step{5.0f};
  VectorInputs inputs{
      view({1, 144, 3}, latent),
      view({1, 256, 4}, text_emb),
      view({1, 50, 256}, style),
      view({1, 1, 3}, latent_mask),
      view({1, 1, 4}, text_mask),
      view({1}, current_step),
      view({1}, total_step)};
  validate_vector_inputs(inputs, metadata);

  VectorInputs invalid = inputs;
  invalid.latent_mask.shape = {1, 1, 2};
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "latent lengths must match");
  invalid = inputs;
  std::vector<float> zeros(3, 0.0f);
  invalid.latent_mask.values = &zeros;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "latent_mask must contain a valid position");
  invalid = inputs;
  std::vector<float> no_text(4, 0.0f);
  invalid.text_mask.values = &no_text;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "text_mask must contain a valid position");
  invalid = inputs;
  std::vector<float> nan_step{std::numeric_limits<float>::quiet_NaN()};
  invalid.current_step.values = &nan_step;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "current_step must be finite");
  invalid = inputs;
  std::vector<float> zero_step{0.0f};
  invalid.total_step.values = &zero_step;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "total_step must be finite and positive");
  invalid = inputs;
  invalid.noisy_latent.shape = {2, 144, 3};
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "batch size must be 1");
  invalid = inputs;
  invalid.text_emb.contiguous = false;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); }, "must be contiguous");
  invalid = inputs;
  invalid.noisy_latent.shape = {1, 144, 0};
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "latent length is outside");
  invalid = inputs;
  invalid.text_emb.shape = {1, 256, 513};
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "text length is outside");
  invalid = inputs;
  invalid.style_ttl.dtype = TensorDtype::Other;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "style_ttl must have dtype float16");
  invalid = inputs;
  std::vector<float> nonfinite_latent = latent;
  nonfinite_latent.front() = std::numeric_limits<float>::infinity();
  invalid.noisy_latent.values = &nonfinite_latent;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "noisy_latent values must be within the finite FP16 range");
  invalid = inputs;
  std::vector<float> infinite_total{std::numeric_limits<float>::infinity()};
  invalid.total_step.values = &infinite_total;
  check_throws(
      [&] { validate_vector_inputs(invalid, metadata); },
      "total_step must be finite and positive");

  bool executor_called = false;
  invalid = inputs;
  invalid.current_step.values = &nan_step;
  check_throws(
      [&] {
        invoke_validated_vector(
            invalid, metadata, [&] { executor_called = true; });
      },
      "current_step must be finite");
  check(!executor_called, "invalid vector input reached executor callback");
  invoke_validated_vector(inputs, metadata, [&] { executor_called = true; });
  check(executor_called, "valid vector input did not reach executor callback");
}

void test_duration_and_trim_helpers() {
  const LatentLayout layout = latent_layout(1.0f, 44100, 512, 6, 512);
  check(layout.waveform_samples == 44100, "waveform sample count mismatch");
  check(layout.latent_length == 15, "latent length mismatch");
  check(
      adjust_duration_for_speed(2.1f, 1.05f) == 2.0f,
      "speed adjustment mismatch");
  check_throws(
      [] { (void)adjust_duration_for_speed(1.0f, 0.0f); },
      "speed must be finite and positive");
  const auto trimmed = trim_waveform({1, 2, 3, 4}, 0.00005f, 44100);
  check(trimmed == std::vector<float>({1, 2}), "waveform trim mismatch");
  const auto combined = combine_vocoder_chunks(
      {{1.0f, 2.0f, 9.0f, 9.0f}, {3.0f, 4.0f, 8.0f}}, {0.3f, 0.2f}, 10, 0.1f);
  check(
      combined == std::vector<float>({1, 2, 9, 9, 0, 3}),
      "multi-chunk full-output concatenation mismatch");
  const std::vector<float> boundary_durations(15, 0.1f);
  check(
      accumulate_chunk_durations(boundary_durations, 0.3f) ==
          5.700000762939453f,
      "duration accumulation must round after every float32-like chunk add");
  const std::vector<std::vector<float>> boundary_waveforms(
      15, std::vector<float>(20000, 1.0f));
  check(
      combine_vocoder_chunks(
          boundary_waveforms, boundary_durations, 44100, 0.3f)
              .size() == 251370,
      "boundary duration accumulation produced the wrong final sample count");
  check_throws(
      [] {
        (void)adjust_duration_for_speed(
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min());
      },
      "adjusted duration is unrepresentable");
  check_throws(
      [] {
        (void)latent_layout(
            std::numeric_limits<float>::max(), 44100, 512, 6, 512);
      },
      "sample count is unrepresentable");
  check_throws(
      [] { (void)latent_layout(1.0f, 44100, 512, 6, 2); },
      "exceeds exported latent bound");
  check_throws(
      [] {
        (void)combine_vocoder_chunks(
            {{1.0f}, {2.0f}},
            {0.1f, 0.1f},
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<float>::max());
      },
      "silence sample count is unrepresentable");
  check_throws(
      [] {
        (void)trim_waveform(
            {1.0f},
            std::numeric_limits<float>::max(),
            std::numeric_limits<int64_t>::max());
      },
      "trim sample count is unrepresentable");
}

void test_wav_writer() {
  const WavLayout layout = validate_wav_layout(5, 44100, 1);
  check(
      layout.data_bytes == 10 && layout.riff_size == 46 &&
          layout.block_align == 2 && layout.byte_rate == 88200,
      "WAV layout mismatch");
  check_throws(
      [] { (void)validate_wav_layout(2, 44100, 0); }, "channels must be in");
  check_throws(
      [] { (void)validate_wav_layout(32768, 44100, 32768); }, "block align");
  check_throws(
      [] { (void)validate_wav_layout(2, std::numeric_limits<int>::max(), 2); },
      "byte rate");
  check_throws(
      [] {
        (void)validate_wav_layout(std::numeric_limits<size_t>::max(), 44100, 1);
      },
      "data size");
  check_throws(
      [] { (void)validate_wav_layout(3, 44100, 2); }, "whole number of frames");
  check(
      !write_pcm16_wav(temp_directory().path("").string(), {0.0f}, 44100),
      "WAV writer accepted a directory as a file");

  const auto path = temp_path("audio.wav");
  check(
      write_pcm16_wav(path.string(), {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, 44100),
      "WAV write failed");
  std::ifstream file(path, std::ios::binary);
  const std::vector<unsigned char> bytes(
      std::istreambuf_iterator<char>(file), {});
  check(bytes.size() == 54, "WAV size mismatch");
  check(
      std::string(bytes.begin(), bytes.begin() + 4) == "RIFF", "missing RIFF");
  check(
      std::string(bytes.begin() + 8, bytes.begin() + 12) == "WAVE",
      "missing WAVE");
  auto u16 = [&](size_t offset) {
    return static_cast<uint16_t>(
        bytes[offset] | (static_cast<uint16_t>(bytes[offset + 1]) << 8));
  };
  auto u32 = [&](size_t offset) {
    return static_cast<uint32_t>(
        bytes[offset] | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
        (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
        (static_cast<uint32_t>(bytes[offset + 3]) << 24));
  };
  check(u32(4) == 46 && u32(40) == 10, "RIFF sizes mismatch");
  check(
      u16(22) == 1 && u32(24) == 44100 && u16(34) == 16, "WAV format mismatch");
  check(
      static_cast<int16_t>(u16(44)) == -32768 &&
          static_cast<int16_t>(u16(52)) == 32767,
      "PCM clipping mismatch");
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 2) {
      throw std::invalid_argument(
          "usage: supertonic_runtime_test [published_voice_style.json]");
    }
    test_preprocessing();
    test_chunking();
    test_style_loading(argc == 2 ? argv[1] : "");
    test_portable_normal_generator();
    test_metadata_contract();
    test_vector_domain_validation();
    test_duration_and_trim_helpers();
    test_wav_writer();
    std::cout << "All Supertonic runtime helper tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << "\n";
    return 1;
  }
}
