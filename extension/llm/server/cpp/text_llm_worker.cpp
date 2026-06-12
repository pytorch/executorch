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
// the stable serving abstraction) — no Python model code, no pybind, no
// in-process Python serving. The OpenAI control plane (Python) spawns this
// process and drives it over JSONL on stdin/stdout (see worker_client.py). The
// JSONL protocol, session management, and the decode loop are shared across all
// workers in worker_loop.h; this file only constructs the engine/tokenizer.
// TextLLMEngine hosts a single session, so the worker serves anonymous requests
// via the shared loop's scratch session and reports no named sessions.

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/server/cpp/worker_loop.h>
#include <executorch/runtime/platform/log.h>

#include <optional>

DEFINE_string(model_path, "", "Self-contained model .pte file path.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");

namespace {
namespace llm = ::executorch::extension::llm;
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

  // The session decodes token ids to text internally; this tokenizer encodes
  // the rendered prompt to ids. Same tokenizer.json -> same vocabulary.
  auto tokenizer = llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer) {
    ET_LOG(Error, "text_llm_worker: failed to load tokenizer");
    return 1;
  }

  return llm::run_worker_stdio_loop(*engine, *tokenizer, engine->metadata());
}
