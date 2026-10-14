/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#if defined(EXECUTORCH_BUILD_MLX)
#include <executorch/examples/models/muse-glimmer/runtime/mlx_executor.h>
#endif
#if defined(EXECUTORCH_BUILD_CUDA)
#include <executorch/examples/models/muse-glimmer/runtime/cuda_backend_factory.h>
#endif

namespace executorch::extension::llm {
runtime::Result<MuseGlimmerBackend> create_muse_glimmer_backend(
    const MuseGlimmerBackendConfig& config) {
#if defined(EXECUTORCH_BUILD_MLX)
  if (config.backend == "mlx") {
    return MuseGlimmerMLXExecutor::create(config);
  }
#endif
#if defined(EXECUTORCH_BUILD_CUDA)
  if (config.backend == "cuda") {
    return create_muse_glimmer_cuda_backend(config);
  }
#endif
  return runtime::Error::NotSupported;
}
} // namespace executorch::extension::llm
