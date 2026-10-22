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
#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>
#endif

#include <memory>
#include <utility>

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
    namespace cb = ::executorch::backends::cuda::batching;
    ET_CHECK_OR_RETURN_ERROR(
        config.max_session_tokens != 1,
        InvalidArgument,
        "Muse Glimmer CUDA: a session needs room for a prompt and prediction");
    ET_CHECK_OR_RETURN_ERROR(
        config.use_dflash || config.dflash_n_draft == 0,
        InvalidArgument,
        "Muse Glimmer CUDA: dflash_n_draft requires DFlash");
    if (config.use_dflash) {
      return MuseGlimmerCudaDFlashExecutor::create(config);
    }
    cb::CudaExecutorConfig executor_config;
    executor_config.model_path = config.model_path;
    if (!config.data_path.empty()) {
      executor_config.data_files.push_back(config.data_path);
    }
    executor_config.max_sessions = config.max_sessions;
    executor_config.max_session_tokens = config.max_session_tokens;
    ET_ASSIGN_OR_RETURN(executor, cb::CudaExecutor::create(executor_config));
    const auto vocab = executor->vocab_size();
    const auto session_tokens = executor->max_session_tokens();
    ET_CHECK_OR_RETURN_ERROR(
        vocab > 200092 && config.bos_id < static_cast<uint64_t>(vocab),
        InvalidProgram,
        "Muse Glimmer CUDA: vocabulary %d cannot hold its special tokens",
        vocab);
    ET_CHECK_OR_RETURN_ERROR(
        session_tokens > 1,
        InvalidArgument,
        "Muse Glimmer CUDA: sessions need room for a prompt and prediction");
    ET_ASSIGN_OR_RETURN(
        spec, make_cuda_preparation(*executor, config, session_tokens));
    return MuseGlimmerBackend{std::move(executor), std::move(spec)};
  }
#endif
  return runtime::Error::NotSupported;
}
} // namespace executorch::extension::llm
