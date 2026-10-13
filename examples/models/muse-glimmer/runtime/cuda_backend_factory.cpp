/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/cuda_backend_factory.h>

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/module/module.h>

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace executorch::extension::llm {
namespace {
namespace cuda_batching = ::executorch::backends::cuda::batching;
using runtime::Error;

constexpr int kKVDtype = static_cast<int>(aten::ScalarType::BFloat16);

// One cell stays reserved as the scratch row that padded static steps write.
int64_t shared_session_tokens(int64_t context, int64_t max_cells, int sessions) {
  return std::min(context, (max_cells - 1) / sessions);
}
} // namespace

runtime::Result<MuseGlimmerBackend> create_muse_glimmer_cuda_backend(
    const MuseGlimmerBackendConfig& config) {
  ET_CHECK_OR_RETURN_ERROR(
      config.backend == "cuda" && !config.model_path.empty() &&
          config.max_sessions > 0 && config.max_session_tokens >= 0 &&
          config.max_session_tokens != 1,
      InvalidArgument,
      "Muse Glimmer CUDA: invalid backend configuration");
  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto module = std::make_unique<Module>(
      config.model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/false);
  ET_CHECK_OK_OR_RETURN_ERROR(module->load());
  ET_ASSIGN_OR_RETURN(context, read_max_context_length(*module));
  ET_ASSIGN_OR_RETURN(vocab, read_vocab_size(*module));
  ET_ASSIGN_OR_RETURN(
      max_cells,
      detail::read_required_positive_int(
          *module, cuda_batching::kMaxCellsMethod));
  ET_CHECK_OR_RETURN_ERROR(
      vocab > 200092 && vocab <= std::numeric_limits<int32_t>::max() &&
          config.bos_id < static_cast<uint64_t>(vocab),
      InvalidProgram,
      "Muse Glimmer CUDA: vocabulary %" PRId64 " cannot hold its special tokens",
      vocab);
  const int64_t session_tokens = config.max_session_tokens
      ? config.max_session_tokens
      : shared_session_tokens(context, max_cells, config.max_sessions);
  ET_CHECK_OR_RETURN_ERROR(
      session_tokens > 1 && session_tokens <= context,
      InvalidArgument,
      "Muse Glimmer CUDA: %" PRId64 " tokens per session for %d sessions does "
      "not fit context %" PRId64 " and %" PRId64 " cells",
      session_tokens,
      config.max_sessions,
      context,
      max_cells);

  auto executor = cuda_batching::CudaExecutor::create(
      std::move(module),
      config.max_sessions,
      static_cast<int>(session_tokens),
      kKVDtype);
  ET_CHECK_OK_OR_RETURN_ERROR(executor.error());
  const auto width = (*executor)->preferred_batch_tokens();
  auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
      MuseGlimmerPreparationSpec{
          aten::ScalarType::BFloat16,
          /*hidden_dim=*/0,
          static_cast<int32_t>(session_tokens),
          static_cast<int32_t>(width),
          static_cast<int32_t>(vocab),
          /*has_vision=*/false,
          /*max_soft_tokens=*/0,
          config.image_limits,
          config.bos_id});
  return MuseGlimmerBackend{std::move(*executor), std::move(spec)};
}

} // namespace executorch::extension::llm
