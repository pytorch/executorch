/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/prompt_preparation.h>
#include <executorch/extension/llm/batching/executor.h>

namespace executorch::extension::llm {

struct MuseGlimmerBackendConfig {
  std::string backend = "mlx";
  std::string model_path;
  std::string data_path;
  std::string pos_embed_path;
  int max_sessions = 4;
  int max_session_tokens = 0; // zero uses published context
  int max_vision_patches = 4096;
  MuseGlimmerImageLimits image_limits;
  batching::Token bos_id = 200000;
};

struct MuseGlimmerBackend {
  std::unique_ptr<batching::Executor> executor;
  std::shared_ptr<const MuseGlimmerPreparationSpec> preparation;
};

runtime::Result<MuseGlimmerBackend> create_muse_glimmer_backend(
    const MuseGlimmerBackendConfig& config);

} // namespace executorch::extension::llm
