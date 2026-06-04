/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic model-execution worker for standard .pte TextLLM models.
//
// All model execution lives here in C++ (via TextLLMEngine / TextLLMSession,
// the stable serving abstraction) — there is no Python model code, no pybind,
// and no in-process Python serving. The OpenAI control plane (Python) spawns
// this process and drives it over JSONL on stdin/stdout (see worker_client.py).
//
// Protocol (one JSON object per line; identical to worker_client.py):
//   worker -> stdout, once:         {"ready": true}
//   client -> stdin,  per request:  {"prompt": str, "max_new_tokens": int,
//                                    "temperature": float}
//   worker -> stdout, per request:  {"token": str} *   (streamed)
//                                   {"done": true, "prompt_tokens": int,
//                                    "completion_tokens": int}
//                               or  {"error": str}
//
// stdout carries ONLY protocol JSON; all logs go to stderr (ET_LOG). One
// request at a time (the control plane serializes; one worker == one session).
//
// V1 cancellation: a request runs to completion (EOS or max_new_tokens); it is
// NOT interruptible mid-generation. The control plane may abandon the response
// on client disconnect, but the worker finishes the in-flight request. (A
// cooperative stop would need a separate control channel; out of scope for V1.)

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

DEFINE_string(model_path, "", "Self-contained model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");

namespace {

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
using json = nlohmann::json;

// Emit one protocol object as a JSON line on stdout. error_handler::replace
// keeps a stray invalid UTF-8 byte (byte-level BPE) from aborting
// serialization.
void emit(const json& obj) {
  std::cout << obj.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
  std::cout.flush();
}

// One generation request: reset the session, encode the prompt, prefill, then
// loop decode_one() streaming complete-UTF-8 text pieces. Mirrors the retired
// Python SessionGenerateAdapter: a terminal step (EOS or stop) ends generation
// and is not emitted or counted.
void handle_request(
    llm::LLMSession& session,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    const json& req) {
  const std::string prompt = req.at("prompt").get<std::string>();
  int64_t max_new = req.value("max_new_tokens", static_cast<int64_t>(-1));
  const float temperature = req.value("temperature", 0.0f);

  if (session.reset() != Error::Ok) {
    throw std::runtime_error("session reset failed");
  }
  // No special tokens: the prompt is already rendered (the control plane
  // applied the chat template), matching the runner's own encode path.
  auto encode_result = tokenizer.encode(prompt, /*bos=*/0, /*eos=*/0);
  if (!encode_result.ok()) {
    throw std::runtime_error("prompt encode failed");
  }
  std::vector<uint64_t> ids = std::move(*encode_result);
  if (ids.empty()) {
    throw std::runtime_error("empty prompt");
  }
  const int64_t num_prompt = static_cast<int64_t>(ids.size());

  if (max_new <= 0) {
    // Fill the remaining context window when the client doesn't bound it.
    const auto it = metadata.find(llm::kMaxContextLen);
    max_new = (it != metadata.end())
        ? std::max<int64_t>(1, it->second - num_prompt)
        : 2048;
  }

  llm::SamplingConfig sampling;
  sampling.temperature = temperature;
  if (session.prefill_tokens(std::move(ids), &sampling) != Error::Ok) {
    throw std::runtime_error("prefill failed");
  }

  std::string buf; // holds bytes not yet forming a complete UTF-8 prefix
  int64_t num_generated = 0;
  for (int64_t step = 0; step < max_new; ++step) {
    auto step_result = session.decode_one(sampling);
    if (step_result.error() != Error::Ok) {
      throw std::runtime_error("decode failed");
    }
    const auto& d = step_result.get();
    if (d.is_terminal) {
      break; // terminal step: not generated output
    }
    ++num_generated;
    buf += d.text_piece;
    const size_t cut = llm::utf8_complete_prefix_len(buf);
    if (cut > 0) {
      emit({{"token", buf.substr(0, cut)}});
      buf.erase(0, cut);
    }
  }
  if (!buf.empty()) {
    emit({{"token", buf}}); // flush any trailing bytes (replaced if incomplete)
  }
  emit(
      {{"done", true},
       {"prompt_tokens", num_prompt},
       {"completion_tokens", num_generated}});
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(
        Error, "text_llm_worker: --model_path and --tokenizer_path required");
    return 1;
  }

  // TextLLMEngine requires a self-contained .pte: external .ptd weights are not
  // supported for shared sessions (a model-specific worker handles that path).
  auto engine = llm::TextLLMEngine::create(
      FLAGS_model_path, FLAGS_tokenizer_path, std::nullopt);
  if (!engine) {
    ET_LOG(Error, "text_llm_worker: failed to create engine");
    return 1;
  }
  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "text_llm_worker: failed to create session");
    return 1;
  }
  auto session = std::move(session_result.get());

  // The session decodes token ids to text internally; this tokenizer encodes
  // the rendered prompt to ids. Same tokenizer.json -> same vocabulary.
  auto tokenizer = llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer) {
    ET_LOG(Error, "text_llm_worker: failed to load tokenizer");
    return 1;
  }
  const auto& metadata = engine->metadata();

  emit({{"ready", true}});

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      handle_request(
          *session, *tokenizer, metadata, nlohmann::json::parse(line));
    } catch (
        const std::exception& e) { // report to the control plane, keep serving
      emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}
