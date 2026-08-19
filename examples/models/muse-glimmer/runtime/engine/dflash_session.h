/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_timing.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_vision_runtime.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::llm {

// Maximum fixed hidden-state input supported by CUDA DFlash artifacts. Each
// artifact records its actual capacity in get_block_size (for example 4 or 16).
inline constexpr int64_t kCudaDFlashHiddenRows = 16;

using DFlashMultimodalSession = MuseGlimmerMultimodalSession;

inline DFlashMultimodalSession* as_dflash_multimodal_session(
    LLMSession* session) {
  return as_muse_glimmer_multimodal_session(session);
}

struct DFlashSessionConfig {
  Module* module = nullptr;
  std::mutex* exec_mutex = nullptr;
  std::atomic<int>* live_sessions = nullptr;
  ::tokenizers::Tokenizer* tokenizer = nullptr;
  std::unordered_map<std::string, int64_t> metadata;
  std::unordered_set<uint64_t> eos_ids;
  MuseGlimmerMutableStateContextOwner* mutable_state = nullptr;
  int session_token = kMuseGlimmerNoMutableSession;
  int64_t max_prefill_chunk = 0;
  int64_t block_size = 0;
  int64_t block_length = 0;
  int64_t n_draft = 0;
  int64_t mask_token_id = 0;
  int64_t n_target_layers = 0;
  int64_t draft_sliding_window = 0;
  int64_t min_target_prefill_chunk = 0;
  int64_t max_target_prefill_chunk = 0;
  int64_t min_draft_prefill_chunk = 0;
  int64_t max_draft_prefill_chunk = 0;
  bool has_draft_prefill = false;
  std::string activation_dtype;
  bool draft_argmax = true;
  bool ignore_eos = false;
  DFlashDecodeTiming* timing = nullptr;
};

::executorch::runtime::Result<std::unique_ptr<LLMSession>>
create_dflash_session(DFlashSessionConfig config);

} // namespace executorch::extension::llm
