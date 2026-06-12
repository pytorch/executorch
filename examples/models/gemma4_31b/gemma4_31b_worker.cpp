/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/models/gemma4_31b/gemma4_31b_engine.h>
#include <executorch/extension/llm/server/cpp/worker_loop.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(data_path, "", "Data file (.ptd) for delegated weights.");
DEFINE_int32(
    max_sessions,
    1,
    "Max physical sessions to host on one weight allocation. CUDA may raise "
    "this when per-session mutable rebinding is available.");
DEFINE_bool(
    warm_resume,
    true,
    "Warm append-only resume for named sessions when the engine supports them.");
DEFINE_int32(bos_id, 2, "BOS token id to prepend to server-rendered prompts.");
DEFINE_int32(eos_id, 1, "EOS token id (Gemma convention: 1).");

namespace {
namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(
        Error, "gemma4_31b_worker: --model_path and --tokenizer_path required");
    return 1;
  }

  llm::Gemma4_31BConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.max_sessions = FLAGS_max_sessions;
  config.eos_id = FLAGS_eos_id;

  auto engine_result = llm::Gemma4_31BEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "gemma4_31b_worker: failed to create engine");
    return 1;
  }
  auto engine = std::move(engine_result.get());

  return llm::run_worker_stdio_loop(
      *engine,
      *engine->tokenizer(),
      engine->metadata(),
      FLAGS_warm_resume,
      {static_cast<uint64_t>(FLAGS_bos_id)});
}
