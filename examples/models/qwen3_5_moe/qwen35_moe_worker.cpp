/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Model-execution worker for Qwen3.5 MoE (CUDA/AOTI).
//
// The Python OpenAI control plane spawns this process and drives it over the
// generic examples/llm_server JSONL worker protocol. This file is intentionally
// model-specific only where it constructs Qwen35MoEEngine.

#include <gflags/gflags.h>

#include <executorch/examples/llm_server/cpp/worker_loop.h>
#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for the CUDA backend.");
DEFINE_int32(
    max_sessions,
    1,
    "Max physical sessions to host on one weight allocation. Clamped to 1 if "
    "the backend cannot isolate per-session mutable state.");
DEFINE_bool(
    warm_resume,
    true,
    "Warm append-only resume for named sessions. Off resets every request.");

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

  return llm::run_worker_stdio_loop(
      *engine, *engine->tokenizer(), engine->metadata(), FLAGS_warm_resume);
}
