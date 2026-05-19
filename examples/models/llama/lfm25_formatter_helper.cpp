/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Persistent companion process for the LFM2.5 formatter model.
//
// Loads an `executorch::extension::llm::TextLLMRunner` once and stays alive,
// reading newline-delimited JSON `format` requests from stdin and writing
// `result`/`status`/`error` messages to stdout. The wire contract is in
// lfm25_formatter_helper_protocol.h.
//
// Built and run by the macOS ExecuWhisper app via `FormatterBridge.swift`,
// which expects the binary at
//   ${EXECUTORCH_PATH}/cmake-out/examples/models/llama/lfm25_formatter_helper
// and the companion shader bundle at
//   $(dirname binary)/mlx.metallib

#include <gflags/gflags.h>

#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>

#include "lfm25_formatter_helper_protocol.h"

DEFINE_string(model_path, "model.pte", "Path to LFM2.5 formatter model (.pte).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.json",
    "Path to the HuggingFace-format tokenizer.json file.");
DEFINE_string(
    tokenizer_config_path,
    "tokenizer_config.json",
    "Path to the HuggingFace-format tokenizer_config.json file (read by the "
    "tokenizers crate when present in the same directory as tokenizer.json; "
    "accepted here for symmetry with FormatterBridge.swift).");
DEFINE_int32(
    default_max_new_tokens,
    256,
    "Fallback max_new_tokens when a request omits it. The Swift bridge always "
    "sets max_new_tokens, so this is mostly a safety net.");

namespace {

namespace fp = lfm25_formatter::helper_protocol;

// Run a single format request through the warm runner. Captures generated
// text via the token callback, captures stats via the stats callback, and
// computes a tokens_per_second figure for the response.
void format_text(
    executorch::extension::llm::TextLLMRunner& runner,
    const std::string& prompt,
    int max_new_tokens,
    double temperature,
    std::string& text_out,
    std::string& stdout_out,
    std::string& stderr_out,
    std::optional<double>& tokens_per_second_out) {
  text_out.clear();
  stdout_out.clear();
  stderr_out.clear();
  tokens_per_second_out.reset();

  // Reset KV cache + stats so each request is independent.
  runner.reset();

  executorch::extension::llm::GenerationConfig config;
  config.echo = false;
  config.ignore_eos = false;
  config.max_new_tokens = max_new_tokens;
  config.temperature = static_cast<float>(temperature);

  std::string accumulated;
  std::optional<executorch::extension::llm::Stats> last_stats;

  // The TextLLMRunner's text generator invokes the token callback for every
  // produced token, including the EOS token (id 7 = "<|im_end|>") that
  // signals end-of-generation. Without filtering, the literal "<|im_end|>"
  // string ends up in the user-visible output. Filter known stop strings
  // here so the rest of the pipeline doesn't have to.
  static const std::vector<std::string> kStopStrings = {
      "<|im_end|>", "<|endoftext|>"};

  // The runner unconditionally prints every generated token and a final
  // PyTorchObserver stats line to stdout (see
  // extension/llm/runner/text_llm_runner.cpp). That conflicts with our
  // JSON-line wire protocol, which also writes to stdout, because the parent
  // process treats every stdout line as a protocol message. Silence stdout
  // for the duration of generate() by redirecting fd 1 to /dev/null, then
  // restore the parent-facing pipe before we emit the protocol response.
  std::cout.flush();
  std::fflush(stdout);
  int saved_stdout_fd = ::dup(STDOUT_FILENO);
  int devnull_fd = ::open("/dev/null", O_WRONLY);
  if (saved_stdout_fd >= 0 && devnull_fd >= 0) {
    ::dup2(devnull_fd, STDOUT_FILENO);
    ::close(devnull_fd);
  }

  const auto err = runner.generate(
      prompt,
      config,
      [&](const std::string& token_text) {
        for (const auto& stop : kStopStrings) {
          if (token_text == stop) {
            return;
          }
        }
        accumulated.append(token_text);
      },
      [&](const executorch::extension::llm::Stats& stats) {
        last_stats.emplace(stats);
      });

  // Restore the parent-facing stdout pipe so subsequent protocol writes
  // (status, result, error) reach the parent process.
  std::fflush(stdout);
  if (saved_stdout_fd >= 0) {
    ::dup2(saved_stdout_fd, STDOUT_FILENO);
    ::close(saved_stdout_fd);
  }

  if (err != ::executorch::runtime::Error::Ok) {
    throw std::runtime_error(
        "TextLLMRunner::generate returned non-Ok error code");
  }

  text_out = std::move(accumulated);

  if (last_stats.has_value()) {
    stdout_out =
        "PyTorchObserver " +
        executorch::extension::llm::stats_to_json_string(*last_stats);

    const long inference_ms =
        last_stats->inference_end_ms - last_stats->inference_start_ms;
    if (inference_ms > 0 && last_stats->num_generated_tokens > 0) {
      tokens_per_second_out = static_cast<double>(
                                  last_stats->num_generated_tokens) *
          1000.0 / static_cast<double>(inference_ms);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // tokenizer_config_path is documented above; reference it so the symbol is
  // not stripped, and so an unsupported value at least surfaces in the log.
  if (!FLAGS_tokenizer_config_path.empty()) {
    ET_LOG(
        Info,
        "Tokenizer config path: %s",
        FLAGS_tokenizer_config_path.c_str());
  }

  try {
    auto tokenizer = ::executorch::extension::llm::load_tokenizer(
        FLAGS_tokenizer_path);
    if (!tokenizer || !tokenizer->is_loaded()) {
      throw std::runtime_error(
          "Failed to load tokenizer: " + FLAGS_tokenizer_path);
    }

    auto runner = ::executorch::extension::llm::create_text_llm_runner(
        FLAGS_model_path, std::move(tokenizer));
    if (!runner) {
      throw std::runtime_error(
          "Failed to construct TextLLMRunner from " + FLAGS_model_path);
    }
    if (runner->load() != ::executorch::runtime::Error::Ok) {
      throw std::runtime_error(
          "TextLLMRunner::load failed for " + FLAGS_model_path);
    }

    if (!fp::write_message(std::cout, fp::encode_ready_message())) {
      std::cerr << "Failed to write helper ready message." << std::endl;
      return 1;
    }

    while (true) {
      fp::Request request;
      std::string request_error;
      if (!fp::read_request(std::cin, &request, &request_error)) {
        if (request_error.empty()) {
          // Clean EOF on stdin — graceful shutdown.
          return 0;
        }
        fp::write_message(
            std::cout,
            fp::encode_error_message(
                std::nullopt,
                "Failed to read helper request",
                request_error));
        return 1;
      }

      if (request.type == fp::Request::Type::Shutdown) {
        return 0;
      }

      const auto& format_request = *request.format;
      try {
        if (format_request.prompt.empty()) {
          throw std::runtime_error("Empty prompt.");
        }

        const int max_new_tokens = format_request.max_new_tokens > 0
            ? format_request.max_new_tokens
            : FLAGS_default_max_new_tokens;

        fp::write_message(
            std::cout,
            fp::encode_status_message(
                format_request.request_id,
                "formatting",
                "Generating formatted text..."));

        std::string text;
        std::string stdout_payload;
        std::string stderr_payload;
        std::optional<double> tokens_per_second;
        format_text(
            *runner,
            format_request.prompt,
            max_new_tokens,
            format_request.temperature,
            text,
            stdout_payload,
            stderr_payload,
            tokens_per_second);

        fp::write_message(
            std::cout,
            fp::encode_result_message(
                format_request.request_id,
                text,
                stdout_payload,
                stderr_payload,
                tokens_per_second));
      } catch (const std::exception& e) {
        fp::write_message(
            std::cout,
            fp::encode_error_message(
                format_request.request_id,
                "Helper formatting failed",
                e.what()));
      }
    }
  } catch (const std::exception& e) {
    fp::write_message(
        std::cout,
        fp::encode_error_message(
            std::nullopt,
            "Failed to start LFM2.5 formatter helper",
            e.what()));
    return 1;
  }
}
