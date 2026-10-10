/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#include <executorch/examples/models/muse-glimmer/runtime/mlx_executor.h>

namespace executorch::extension::llm {
runtime::Result<MuseGlimmerBackend> create_muse_glimmer_backend(
    const MuseGlimmerBackendConfig& config) {
  if (config.backend != "mlx") {
    return runtime::Error::NotSupported;
  }
  return MuseGlimmerMLXExecutor::create(config);
}
} // namespace executorch::extension::llm
