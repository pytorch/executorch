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
    auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
        MuseGlimmerPreparationSpec{
            aten::ScalarType::BFloat16,
            /*hidden_dim=*/0,
            session_tokens,
            static_cast<int32_t>(executor->preferred_batch_tokens()),
            vocab,
            /*has_vision=*/false,
            /*max_soft_tokens=*/0,
            config.image_limits,
            config.bos_id});
    return MuseGlimmerBackend{std::move(executor), std::move(spec)};
  }
#endif
  return runtime::Error::NotSupported;
}
} // namespace executorch::extension::llm
