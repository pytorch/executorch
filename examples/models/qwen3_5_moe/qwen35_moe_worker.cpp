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
// Multi-session: the engine loads weights once and hosts multiple isolated
// sessions on that one ~18GB allocation; the shared worker loop (worker_loop.h)
// routes requests to per-session_id state (up to --max_sessions) and warm-
// resumes each session's context across requests (append-only suffix prefill).
// Execution is synchronous (one in-flight request).

#include <gflags/gflags.h>

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/extension/llm/server/cpp/worker_loop.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for the CUDA backend.");
DEFINE_int32(
    max_sessions,
    1,
    "Max physical sessions to host on the one weight allocation (CUDA "
    "per-session mutable rebinding). Clamped to 1 if the backend cannot "
    "rebind.");
DEFINE_bool(
    warm_resume,
    true,
    "Warm append-only resume for named sessions: prefill only the suffix when a "
    "request's tokens extend the session's resident context. Off resets every "
    "request (useful for A/B measurement).");

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
  config.max_sessions = FLAGS_max_sessions;

  auto engine_result = llm::Qwen35MoEEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "qwen35_moe_worker: failed to create engine");
    return 1;
  }
  auto engine = std::move(engine_result.get());

  // The engine's tokenizer encodes the rendered prompt to ids; sessions decode
  // ids back to text internally. The shared loop owns per-session_id state.
  ::tokenizers::Tokenizer* tokenizer = engine->tokenizer();

  return llm::run_worker_stdio_loop(
      *engine, *tokenizer, engine->metadata(), FLAGS_warm_resume);
}
