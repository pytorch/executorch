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
#if defined(EXECUTORCH_BUILD_NATIVE)
#include <executorch/backends/native/batching/native_executor.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
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
#if defined(EXECUTORCH_BUILD_NATIVE)
  if (config.backend == "native") {
    // Needs a package exported with export_hf_llm.py --offgraph_cache
    // --logits_to_keep selected; text only.
    ET_CHECK_OR_RETURN_ERROR(
        !config.use_dflash,
        NotSupported,
        "Muse Glimmer native: DFlash is unsupported");
    ET_CHECK_OR_RETURN_ERROR(
        (config.native_device == "cpu" || config.native_device == "gpu") &&
            (config.native_kv_dtype == "fp32" ||
             config.native_kv_dtype == "fp16"),
        InvalidArgument,
        "Muse Glimmer native: --native_device must be cpu or gpu and "
        "--native_kv_dtype fp32 or fp16");
    ET_CHECK_OR_RETURN_ERROR(
        config.data_path.empty() && config.pos_embed_path.empty(),
        InvalidArgument,
        "Muse Glimmer native: the .ptn holds the weights and serves text "
        "only; omit --data_path and --pos_embed_path");
    ptn::register_ggml_cache_builders();
    const ptn::GgmlDevice device = config.native_device == "cpu"
        ? ptn::GgmlDevice::Cpu
        : ptn::GgmlDevice::Gpu;
    ET_ASSIGN_OR_RETURN(
        executor,
        ptn::NativeExecutor::create(
            {.model_path = config.model_path,
             .make_host =
                 [device] {
                   return ptn::GgmlEngineHost::create({.device = device});
                 },
             .cache_backend_id = ptn::kGgmlCacheBackendId,
             .max_sessions = config.max_sessions,
             .max_session_tokens = config.max_session_tokens,
             .kv_dtype = config.native_kv_dtype == "fp16"
                 ? ptn::ScalarType::Half
                 : ptn::ScalarType::Float}));
    const int32_t vocab = executor->vocab_size();
    ET_CHECK_OR_RETURN_ERROR(
        vocab > 200092 && config.bos_id < static_cast<uint64_t>(vocab),
        InvalidProgram,
        "Muse Glimmer native: vocabulary %d cannot hold its special tokens",
        vocab);
    auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
        MuseGlimmerPreparationSpec{
            aten::ScalarType::Float,
            /*hidden_dim=*/0,
            executor->max_session_tokens(),
            static_cast<int32_t>(executor->preferred_batch_tokens()),
            vocab,
            /*has_vision=*/false,
            /*max_soft_tokens=*/0,
            config.image_limits,
            config.bos_id});
    return MuseGlimmerBackend{std::move(executor), std::move(spec)};
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
