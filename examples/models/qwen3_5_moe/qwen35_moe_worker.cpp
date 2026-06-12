/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Model-execution worker for Qwen3.5 MoE (CUDA/AOTI).
//
// All model execution lives here in C++ via Qwen35MoEEngine / LLMSession; no
// Python model code, no pybind. The OpenAI control plane (serve.py) spawns this
// process and drives it over JSONL through the generic WorkerClient — the same
// protocol and decode loop every worker uses (worker_loop.h); this file only
// constructs the engine/session.
//
// Isolation rationale: executing the AOTI CUDA model inside a live asyncio HTTP
// process segfaults in the int4 matmul (validated). Here the model runs in a
// plain synchronous loop in its own process, which is reliable.
//
// V1: single-slot (one engine == one ~18GB weight allocation == one session);
// the control plane queues concurrent requests on the resident session.

#include <gflags/gflags.h>

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/server/cpp/worker_loop.h>
#include <executorch/runtime/platform/log.h>

#include <utility>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for the CUDA backend.");
DEFINE_bool(cuda_graph, false, "Enable CUDA graph for the decode method.");

namespace {
namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
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

  return llm::run_worker_stdio_loop(*session, *tokenizer, engine->metadata());
}
