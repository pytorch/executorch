/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "supertonic_runner.h"
#include "wav_writer.h"

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

DEFINE_string(pte, "", "Path to the single Supertonic FP16 MLX .pte file.");
DEFINE_string(
    asset_dir,
    "",
    "Published Supertonic asset root containing onnx/unicode_indexer.json.");
DEFINE_string(
    voice_style,
    "",
    "Exactly one published batch-1 voice-style JSON path.");
DEFINE_string(text, "", "Text to synthesize.");
DEFINE_string(language, "en", "Published Supertonic language tag.");
DEFINE_double(speed, 1.05, "Positive duration speed divisor.");
DEFINE_uint64(seed, 42, "Portable xorshift64/Box-Muller latent seed.");
DEFINE_string(output, "supertonic.wav", "Output mono PCM16 WAV path.");
DEFINE_bool(
    server_jsonl,
    false,
    "Keep the model resident and serve JSONL requests on stdin.");
DEFINE_string(
    warmup_text,
    "Warmup.",
    "Text synthesized and discarded before the JSONL ready event.");

namespace {

using Json = nlohmann::ordered_json;

constexpr size_t kMaxJsonLineBytes = 64 * 1024;

void require_file(const std::filesystem::path& path, const char* flag) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument(
        std::string(flag) + " does not name a readable file: " + path.string());
  }
}

void write_json_line(const Json& value) {
  std::cout << value.dump() << '\n';
  std::cout.flush();
  if (!std::cout) {
    throw std::runtime_error("failed to write JSONL response");
  }
}

void write_error(std::optional<uint64_t> id, const std::string& message) {
  write_json_line({
      {"type", "error"},
      {"id", id.has_value() ? Json(*id) : Json(nullptr)},
      {"message", message},
  });
}

double elapsed_seconds(std::chrono::steady_clock::time_point started) {
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (!std::isfinite(elapsed) || elapsed < 0.0) {
    throw std::runtime_error("elapsed time is not finite and nonnegative");
  }
  return elapsed;
}

std::optional<std::string> validate_output_path(
    const std::filesystem::path& output) {
  std::error_code error;
  if (std::filesystem::exists(output, error)) {
    return "output already exists: " + output.string();
  }
  if (error) {
    return "failed to inspect output path: " + output.string();
  }
  const std::filesystem::path parent = output.parent_path().empty()
      ? std::filesystem::path(".")
      : output.parent_path();
  if (!std::filesystem::is_directory(parent, error)) {
    return "output parent is not an existing directory: " + parent.string();
  }
  if (error) {
    return "failed to inspect output parent: " + parent.string();
  }
  return std::nullopt;
}

enum class LineReadResult {
  Line,
  End,
  TooLong,
};

LineReadResult read_json_line(std::istream& input, std::string& line) {
  line.clear();
  char character;
  while (input.get(character)) {
    if (character == '\n') {
      return LineReadResult::Line;
    }
    if (line.size() == kMaxJsonLineBytes) {
      while (input.get(character) && character != '\n') {
      }
      return LineReadResult::TooLong;
    }
    line.push_back(character);
  }
  return line.empty() ? LineReadResult::End : LineReadResult::Line;
}

void write_wav_without_overwrite(
    const std::filesystem::path& output,
    const std::vector<float>& waveform,
    int sample_rate) {
  const std::filesystem::path parent = output.parent_path().empty()
      ? std::filesystem::path(".")
      : output.parent_path();
  std::string temporary_template =
      (parent / ("." + output.filename().string() + ".XXXXXX")).string();
  int temporary_fd = mkstemp(temporary_template.data());
  if (temporary_fd < 0) {
    throw std::runtime_error(
        "failed to create temporary WAV: " + output.string() + ": " +
        std::strerror(errno));
  }
  const std::filesystem::path temporary(temporary_template);
  std::error_code cleanup_error;
  try {
    const std::string fd_path = "/dev/fd/" + std::to_string(temporary_fd);
    if (!supertonic::write_pcm16_wav(fd_path, waveform, sample_rate)) {
      throw std::runtime_error(
          "failed to write output WAV: " + output.string());
    }
    if (close(temporary_fd) != 0) {
      const int close_error = errno;
      temporary_fd = -1;
      throw std::runtime_error(
          "failed to close output WAV: " + output.string() + ": " +
          std::strerror(close_error));
    }
    temporary_fd = -1;
    if (renamex_np(temporary.c_str(), output.c_str(), RENAME_EXCL) != 0) {
      const int rename_error = errno;
      if (rename_error == EEXIST) {
        throw std::invalid_argument(
            "output already exists: " + output.string());
      }
      throw std::runtime_error(
          "failed to finalize output WAV: " + output.string() + ": " +
          std::strerror(rename_error));
    }
  } catch (...) {
    if (temporary_fd >= 0) {
      close(temporary_fd);
    }
    std::filesystem::remove(temporary, cleanup_error);
    throw;
  }
}

int run_jsonl_server(
    supertonic::SupertonicRunner& runner,
    const supertonic::SynthesisOptions& defaults,
    double load_seconds) {
  supertonic::SynthesisOptions warmup = defaults;
  warmup.text = FLAGS_warmup_text;
  const auto warmup_started = std::chrono::steady_clock::now();
  (void)runner.synthesize(warmup);
  write_json_line({
      {"type", "ready"},
      {"protocol_version", 1},
      {"sample_rate", runner.metadata().sample_rate},
      {"load_seconds", load_seconds},
      {"warmup_seconds", elapsed_seconds(warmup_started)},
  });

  uint64_t last_request_id = 0;
  std::string line;
  while (true) {
    const LineReadResult line_result = read_json_line(std::cin, line);
    if (line_result == LineReadResult::End) {
      return 0;
    }
    if (line_result == LineReadResult::TooLong) {
      write_error(std::nullopt, "JSON request exceeds 65536 bytes");
      continue;
    }
    std::optional<uint64_t> id;
    try {
      const Json request = Json::parse(line);
      if (!request.is_object() || !request.contains("type") ||
          !request["type"].is_string()) {
        throw std::invalid_argument("request must be an object with type");
      }
      const std::string type = request["type"].get<std::string>();
      if (type == "shutdown") {
        if (request.size() != 1) {
          throw std::invalid_argument("shutdown request has unexpected fields");
        }
        write_json_line({{"type", "stopped"}});
        return 0;
      }
      if (type != "synthesize") {
        throw std::invalid_argument("unknown request type: " + type);
      }
      if (!request.contains("id") || !request["id"].is_number_unsigned() ||
          request["id"].get<uint64_t>() == 0) {
        throw std::invalid_argument("id must be a positive integer");
      }
      id = request["id"].get<uint64_t>();
      static const std::set<std::string> expected{
          "type", "id", "text", "output"};
      std::set<std::string> actual;
      for (auto item = request.begin(); item != request.end(); ++item) {
        actual.insert(item.key());
      }
      if (actual != expected) {
        throw std::invalid_argument("synthesize request has unexpected fields");
      }
      if (!request["text"].is_string() ||
          request["text"].get_ref<const std::string&>().empty()) {
        throw std::invalid_argument("text must be a nonempty string");
      }
      if (!request["output"].is_string() ||
          request["output"].get_ref<const std::string&>().empty()) {
        throw std::invalid_argument("output must be a nonempty string");
      }
      if (*id <= last_request_id) {
        throw std::invalid_argument(
            "id must increase monotonically above " +
            std::to_string(last_request_id));
      }
      last_request_id = *id;

      const std::filesystem::path output = request["output"].get<std::string>();
      if (const auto error = validate_output_path(output)) {
        throw std::invalid_argument(*error);
      }
      supertonic::SynthesisOptions options = defaults;
      options.text = request["text"].get<std::string>();
      const auto result = runner.synthesize(options);
      write_wav_without_overwrite(
          output,
          result.waveform,
          static_cast<int>(runner.metadata().sample_rate));
      const double audio_seconds = static_cast<double>(result.waveform.size()) /
          runner.metadata().sample_rate;
      write_json_line({
          {"type", "result"},
          {"id", *id},
          {"output", output.string()},
          {"samples", result.waveform.size()},
          {"audio_seconds", audio_seconds},
          {"synthesis_seconds", result.elapsed_seconds},
          {"rtf",
           audio_seconds > 0.0 ? result.elapsed_seconds / audio_seconds : 0.0},
      });
    } catch (const Json::parse_error&) {
      write_error(std::nullopt, "malformed JSON request");
    } catch (const std::exception& error) {
      std::cerr << "Supertonic JSONL request failed: " << error.what() << '\n';
      write_error(id, error.what());
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Synthesize Supertonic speech with a batch-1 FP16 MLX PTE.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    if (FLAGS_pte.empty() || FLAGS_asset_dir.empty() ||
        FLAGS_voice_style.empty() ||
        (!FLAGS_server_jsonl && FLAGS_text.empty())) {
      throw std::invalid_argument(
          "--pte, --asset_dir, and --voice_style are required; --text is "
          "required unless --server_jsonl is set");
    }
    if (FLAGS_server_jsonl && FLAGS_warmup_text.empty()) {
      throw std::invalid_argument("--warmup_text must not be empty");
    }
    const std::filesystem::path pte(FLAGS_pte);
    const std::filesystem::path indexer =
        std::filesystem::path(FLAGS_asset_dir) / "onnx" /
        "unicode_indexer.json";
    require_file(pte, "--pte");
    require_file(indexer, "--asset_dir");
    const std::string style =
        supertonic::require_single_voice_style_path({FLAGS_voice_style});
    supertonic::validate_language(FLAGS_language);
    if (!std::isfinite(FLAGS_speed) || FLAGS_speed <= 0.0 ||
        FLAGS_speed > std::numeric_limits<float>::max()) {
      throw std::invalid_argument(
          "--speed must be finite, positive, and representable as float");
    }
    require_file(style, "--voice_style");

    const auto load_started = std::chrono::steady_clock::now();
    supertonic::SupertonicRunner runner(pte.string(), indexer.string());
    const double load_seconds = elapsed_seconds(load_started);

    supertonic::SynthesisOptions options;
    options.text = FLAGS_text;
    options.language = FLAGS_language;
    options.voice_style_paths = {style};
    options.speed = static_cast<float>(FLAGS_speed);
    options.seed = FLAGS_seed;
    if (FLAGS_server_jsonl) {
      return run_jsonl_server(runner, options, load_seconds);
    }

    auto result = runner.synthesize(options);
    if (!supertonic::write_pcm16_wav(
            FLAGS_output,
            result.waveform,
            static_cast<int>(runner.metadata().sample_rate))) {
      throw std::runtime_error("failed to write output WAV: " + FLAGS_output);
    }
    const double audio_seconds = static_cast<double>(result.waveform.size()) /
        runner.metadata().sample_rate;
    std::cout << "Wrote " << result.waveform.size() << " samples ("
              << audio_seconds << " s) at " << runner.metadata().sample_rate
              << " Hz to " << FLAGS_output << "\n";
    if (audio_seconds > 0.0) {
      std::cout << "Synthesis " << result.elapsed_seconds << " s, RTF "
                << result.elapsed_seconds / audio_seconds << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Supertonic synthesis failed: " << error.what() << "\n";
    return 1;
  }
}
