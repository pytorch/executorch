// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Run a Whisper PTN (encoder + cross_kv + decoder methods, see
// export_hf_whisper) on a native engine: encode a raw fp32 log-mel
// window once, project cross-attention K/V once, prefill the forced
// decoder prefix into the static KV cache, then greedily decode one token
// per call, advancing cache_position. Cross K/V are set as decoder inputs
// once before prefill; input staging persists across execute() calls, so
// decode steps only set the token and its position. Optionally compare
// the id sequence with an eager reference. The decoder executable's cache
// buffers are process-lifetime state: one run decodes one utterance.
// Exit codes: 0 success, 1 invalid arguments, 2 execution error, 3 mismatch.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/backends/native/examples/audio/WhisperRunner.h>
#include <executorch/backends/native/examples/runner_utils/runner_utils.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <gflags/gflags.h>

DEFINE_string(package, "", "Path to the *.ptn to run. Required.");
DEFINE_string(input, "", "Log-mel input tensor, raw float32. Required.");
DEFINE_string(prompt, "", "Forced decoder prefix ids, one per line. Required.");
DEFINE_string(
    expected,
    "",
    "Reference id sequence to compare against, one per line.");
DEFINE_string(eos_ids, "", "Comma-separated end-of-sequence ids. Required.");
DEFINE_string(encoder_method, "encoder", "Encoder method to run.");
DEFINE_string(
    cross_kv_method,
    "cross_kv",
    "Cross-KV projection method to run.");
DEFINE_string(decoder_method, "decoder", "Decoder method to run.");
DEFINE_int64(max_new_tokens, 32, "Stop after generating this many tokens.");

namespace {

int64_t parse_id(const std::string& token) {
  size_t consumed = 0;
  const int64_t id = std::stoll(token, &consumed);
  if (token.find_first_not_of(" \t\r\n", consumed) != std::string::npos) {
    throw std::invalid_argument("not an integer id: " + token);
  }
  return id;
}

std::unordered_set<int64_t> parse_id_list(const std::string& flag) {
  std::unordered_set<int64_t> ids;
  size_t start = 0;
  while (start <= flag.size()) {
    const size_t comma = flag.find(',', start);
    const std::string token =
        flag.substr(start, comma == std::string::npos ? comma : comma - start);
    if (!token.empty()) {
      ids.insert(parse_id(token));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return ids;
}

std::vector<int64_t> read_ids(const std::string& path) {
  std::vector<int64_t> ids;
  for (const std::string& line : ptn::runner_utils::read_lines(path)) {
    if (!line.empty()) {
      ids.push_back(parse_id(line));
    }
  }
  return ids;
}

void print_ids(const char* label, const std::vector<int64_t>& ids) {
  std::printf("%s (%zu):", label, ids.size());
  for (const int64_t id : ids) {
    std::printf(" %lld", static_cast<long long>(id));
  }
  std::printf("\n");
}

} // namespace

namespace ptn {

int run_whisper(const std::function<std::unique_ptr<EngineHost>()>& make_host) {
  if (FLAGS_package.empty() || FLAGS_input.empty() || FLAGS_prompt.empty() ||
      FLAGS_eos_ids.empty() || FLAGS_max_new_tokens < 1) {
    std::fprintf(
        stderr,
        "error: --package, --input, --prompt, --eos_ids and a positive "
        "--max_new_tokens are required\n");
    return 1;
  }

  std::unordered_set<int64_t> eos_ids;
  std::vector<int64_t> ids;
  std::vector<int64_t> expected;
  try {
    eos_ids = parse_id_list(FLAGS_eos_ids);
    if (eos_ids.empty()) {
      std::fprintf(stderr, "error: --eos_ids holds no ids\n");
      return 1;
    }
    ids = read_ids(FLAGS_prompt);
    if (ids.empty()) {
      std::fprintf(stderr, "error: --prompt holds no ids\n");
      return 1;
    }

    if (!FLAGS_expected.empty()) {
      expected = read_ids(FLAGS_expected);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: invalid token ids: %s\n", e.what());
    return 1;
  }

  try {
    const auto package =
        std::make_shared<const ptn::Package>(ptn::Package::load(FLAGS_package));
    const ptn::ByteSpan program_bytes = package->program_bytes();
    const auto program = std::make_shared<const ptn::Program>(
        ptn::Program::load(program_bytes.data(), program_bytes.size()));
    std::printf(
        "package:   %s (program %zu bytes, %zu constants / %zu bytes)\n",
        FLAGS_package.c_str(),
        program_bytes.size(),
        package->keys().size(),
        package->constant_bytes());

    const std::unique_ptr<ptn::EngineHost> host = make_host();
    if (!host) {
      throw std::runtime_error("failed to create engine host");
    }
    const std::unique_ptr<ptn::EngineContext> context =
        host->create_context(program, package);
    if (!context) {
      throw std::runtime_error("failed to create engine context");
    }
    std::printf(
        "engine:    %s on %s\n",
        host->name().c_str(),
        host->device_name().c_str());

    const std::unique_ptr<ptn::EngineExecutable> encoder =
        context->compile(FLAGS_encoder_method);
    const std::unique_ptr<ptn::EngineExecutable> cross_kv =
        context->compile(FLAGS_cross_kv_method);
    const std::unique_ptr<ptn::EngineExecutable> decoder =
        context->compile(FLAGS_decoder_method);
    if (!encoder || !cross_kv || !decoder) {
      throw std::runtime_error("failed to compile Whisper methods");
    }
    std::printf(
        "methods:   %s (%zu in, %zu out), %s (%zu in, %zu out), "
        "%s (%zu in, %zu out)\n",
        FLAGS_encoder_method.c_str(),
        encoder->num_inputs(),
        encoder->num_outputs(),
        FLAGS_cross_kv_method.c_str(),
        cross_kv->num_inputs(),
        cross_kv->num_outputs(),
        FLAGS_decoder_method.c_str(),
        decoder->num_inputs(),
        decoder->num_outputs());
    std::printf("constants: %zu package bytes\n", package->constant_bytes());

    const size_t num_cross = cross_kv->num_outputs();
    if (encoder->num_inputs() != 1 || encoder->num_outputs() != 1 ||
        cross_kv->num_inputs() != 1 || num_cross == 0 || num_cross % 2 != 0 ||
        decoder->num_inputs() != 2 + num_cross || decoder->num_outputs() != 1) {
      std::fprintf(
          stderr,
          "error: expected encoder 1->1, cross_kv 1->%zu and decoder "
          "%zu->1\n",
          num_cross,
          2 + num_cross);
      return 2;
    }
    bool dtypes_ok = encoder->input_dtype(0) == ptn::kFloat &&
        encoder->output_dtype(0) == ptn::kFloat &&
        cross_kv->input_dtype(0) == ptn::kFloat &&
        decoder->input_dtype(0) == ptn::kLong &&
        decoder->input_dtype(1) == ptn::kLong &&
        decoder->output_dtype(0) == ptn::kFloat;
    for (size_t i = 0; i < num_cross; ++i) {
      dtypes_ok = dtypes_ok && cross_kv->output_dtype(i) == ptn::kFloat &&
          decoder->input_dtype(2 + i) == ptn::kFloat &&
          cross_kv->output_sizes(i) == decoder->input_sizes(2 + i);
    }
    if (!dtypes_ok) {
      std::fprintf(
          stderr,
          "error: expected fp32 mel/hidden/cross/logits and int64 "
          "ids/cache_position, with cross shapes matching decoder inputs\n");
      return 2;
    }
    const int64_t cache_capacity = decoder->input_sizes(1).front();
    if (static_cast<int64_t>(ids.size()) > cache_capacity) {
      std::fprintf(
          stderr,
          "error: prompt has %zu ids but the KV cache holds %lld\n",
          ids.size(),
          static_cast<long long>(cache_capacity));
      return 1;
    }

    const std::vector<int64_t> mel_sizes = encoder->input_sizes(0);
    const std::vector<float> mel = ptn::runner_utils::read_floats(FLAGS_input);
    if (static_cast<int64_t>(mel.size()) !=
        ptn::runner_utils::numel_of(mel_sizes)) {
      std::fprintf(
          stderr,
          "error: input has %zu floats, the encoder wants %lld %s\n",
          mel.size(),
          static_cast<long long>(ptn::runner_utils::numel_of(mel_sizes)),
          ptn::runner_utils::shape_str(mel_sizes).c_str());
      return 2;
    }
    encoder->set_input(0, mel.data(), mel.size(), ptn::kFloat);
    encoder->execute();

    const std::vector<int64_t> hidden_sizes = encoder->output_sizes(0);
    std::vector<float> hidden(
        static_cast<size_t>(ptn::runner_utils::numel_of(hidden_sizes)));
    encoder->get_output(0, hidden.data(), hidden.size(), ptn::kFloat);
    std::printf(
        "encoder:   %s -> %s\n",
        ptn::runner_utils::shape_str(mel_sizes).c_str(),
        ptn::runner_utils::shape_str(hidden_sizes).c_str());
    if (hidden_sizes != cross_kv->input_sizes(0)) {
      std::fprintf(
          stderr,
          "error: encoder output %s does not match cross_kv input %s\n",
          ptn::runner_utils::shape_str(hidden_sizes).c_str(),
          ptn::runner_utils::shape_str(cross_kv->input_sizes(0)).c_str());
      return 2;
    }

    // Project cross-attention K/V once and stage them as decoder inputs.
    // Staging persists across execute() calls, so decode steps below only
    // set the token id and cache_position.
    cross_kv->set_input(0, hidden.data(), hidden.size(), ptn::kFloat);
    cross_kv->execute();
    for (size_t i = 0; i < num_cross; ++i) {
      std::vector<float> cross(static_cast<size_t>(
          ptn::runner_utils::numel_of(cross_kv->output_sizes(i))));
      cross_kv->get_output(i, cross.data(), cross.size(), ptn::kFloat);
      decoder->set_input(2 + i, cross.data(), cross.size(), ptn::kFloat);
    }
    std::printf(
        "cross_kv:  %zu tensors of %s\n",
        num_cross,
        ptn::runner_utils::shape_str(cross_kv->output_sizes(0)).c_str());

    print_ids("prompt", ids);
    const auto decode_step = [&decoder_executable = *decoder](
                                 const std::vector<int64_t>& window_ids,
                                 int64_t first_position) {
      std::vector<int64_t> positions(window_ids.size());
      for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = first_position + static_cast<int64_t>(i);
      }
      decoder_executable.resize_input(
          0, {1, static_cast<int64_t>(window_ids.size())});
      decoder_executable.resize_input(
          1, {static_cast<int64_t>(positions.size())});
      decoder_executable.set_input(
          0, window_ids.data(), window_ids.size(), ptn::kLong);
      decoder_executable.set_input(
          1, positions.data(), positions.size(), ptn::kLong);
      decoder_executable.execute();

      const std::vector<int64_t> logit_sizes =
          decoder_executable.output_sizes(0);
      const int64_t vocab = logit_sizes.back();
      std::vector<float> logits(
          static_cast<size_t>(ptn::runner_utils::numel_of(logit_sizes)));
      decoder_executable.get_output(
          0, logits.data(), logits.size(), ptn::kFloat);
      const size_t last_row = logits.size() - static_cast<size_t>(vocab);
      const std::vector<float> row(
          logits.begin() + static_cast<std::ptrdiff_t>(last_row), logits.end());
      return static_cast<int64_t>(ptn::runner_utils::top_k(row, 1).front());
    };

    const int64_t max_new_tokens = std::min(
        FLAGS_max_new_tokens,
        cache_capacity - static_cast<int64_t>(ids.size()));
    if (max_new_tokens > 0) {
      int64_t next = decode_step(ids, 0);
      for (int64_t step = 0; step < max_new_tokens; ++step) {
        ids.push_back(next);
        if (eos_ids.count(next) != 0 || step + 1 == max_new_tokens) {
          break;
        }
        next = decode_step({next}, static_cast<int64_t>(ids.size()) - 1);
      }
    }
    print_ids("transcript", ids);

    if (FLAGS_expected.empty()) {
      std::printf("result:    ran (no reference given)\n");
      return 0;
    }
    const bool pass = ids == expected;
    if (!pass) {
      print_ids("expected", expected);
    }
    std::printf("result:    %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

} // namespace ptn
