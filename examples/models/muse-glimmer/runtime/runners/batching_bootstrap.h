/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/tokenizer.h>

DECLARE_int32(max_inflight_requests);
DECLARE_uint64(max_input_frame_bytes);

namespace executorch::extension::llm {

// Shared startup only: execution and scheduling remain in the generic runtime.
struct MuseGlimmerBatchingRuntime {
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  MuseGlimmerBackend backend;
  std::unique_ptr<serving::ServingRuntime> runtime;
};
std::unique_ptr<MuseGlimmerBatchingRuntime>
create_muse_glimmer_batching_runtime();

} // namespace executorch::extension::llm
