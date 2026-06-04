/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Model-execution worker for Qwen3.5 MoE (CUDA/AOTI).
//
// All model execution lives here in C++ via Qwen35MoEEngine / LLMSession; there
// is no Python model code and no pybind. The OpenAI control plane (serve.py)
// spawns this process and drives it over JSONL on stdin/stdout through the
// generic WorkerClient (extension/llm/server/python/worker_client.py) — the
// same protocol the generic text_llm_worker speaks.
//
// Isolation rationale: executing the AOTI CUDA model inside a live asyncio HTTP
// process segfaults in the int4 matmul (validated). Here the model runs in a
// plain synchronous loop in its own process, which is reliable.
//
// Protocol (one JSON object per line):
//   worker -> stdout, once:         {"ready": true}
//   client -> stdin,  per request:  {"prompt": str, "max_new_tokens": int,
//                                    "temperature": float}
//   worker -> stdout, per request:  {"token": str} *   (streamed)
//                                   {"done": true, "prompt_tokens": int,
//                                    "completion_tokens": int}
//                               or  {"error": str}
//
// stdout carries ONLY protocol JSON; all logs go to stderr (ET_LOG).
//
// V1: single-slot (one engine == one ~18GB weight allocation == one session);
// the control plane queues concurrent requests on the resident session. A
// request runs to completion (EOS or max_new_tokens) and is NOT interruptible
// mid-generation; the control plane may abandon the response on disconnect, but
// the worker finishes the in-flight request.

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for the CUDA backend.");
DEFINE_bool(cuda_graph, false, "Enable CUDA graph for the decode method.");

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
// loop decode_one() streaming complete-UTF-8 text pieces. A terminal step (EOS
// or stop) ends generation and is not emitted or counted.
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
  // applied the chat template).
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
        Error, "qwen35_moe_worker: --model_path and --tokenizer_path required");
    return 1;
  }

  llm::Qwen35MoEConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.cuda_graph = FLAGS_cuda_graph;

  auto engine_result = llm::Qwen35MoEEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "qwen35_moe_worker: failed to create engine");
    return 1;
  }
  auto engine = std::move(engine_result.get());

  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "qwen35_moe_worker: failed to create session");
    return 1;
  }
  auto session = std::move(session_result.get());

  // The engine's tokenizer encodes the rendered prompt to ids; the session
  // decodes ids back to text internally.
  ::tokenizers::Tokenizer* tokenizer = engine->tokenizer();
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
    } catch (const std::exception& e) { // report and keep serving
      emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}
