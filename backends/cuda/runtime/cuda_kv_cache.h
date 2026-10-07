/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::backends::cuda {

namespace cache = ::executorch::extension::llm::cache;

struct OffGraphKVMetrics {
  int64_t logical_length{0};
  // Rows currently allocated for each flat layer.
  int64_t flat_capacity{0};
  // Times flat storage has been reallocated. Every growth moves the storage,
  // so this also identifies the addresses a compiled program was bound to.
  int64_t growth_count{0};
  int64_t allocated_bytes{0};
};

// Backend face of the off-graph KV cache: what the CUDA delegate needs from
// whatever cache the runner installed under its registry key.
//
// Named here rather than in cache.h for the same reason as MLXCache: a backend
// face speaks the backend's own types (here, AOTI delegate handles and CUDA
// streams) and the neutral header cannot know about them.
//
// The cache owns memory only. Lowering replaced kvcache::update_and_attend
// with index_copy_ + triton::sdpa over sequence-major (BSHD) storage declared
// at the maximum capacity. Every access is bounded by kv_len along the
// sequence, so a flat layer may be backed by fewer rows than declared and grow
// on demand; the cache points the program's storage constants at its current
// allocations.
class CudaKVCache {
 public:
  static constexpr const char* kFaceName = "cuda.CudaKVCache";

  virtual ~CudaKVCache() = default;

  // Load time. Discovers this program's KV storage constants. One cache serves
  // every method of a model, so this is called once per delegate handle.
  //
  // Answers whether this program carries off-graph storage at all: a method
  // that does not (an embedding or vision pass) has no use for the cache, and
  // the caller should stop associating it with one.
  virtual runtime::Result<bool> note_handle(CudaDelegateHandle* handle) = 0;
  virtual void forget_handle(CudaDelegateHandle* handle) = 0;

  // Execute time, after prepare_step. Points the program's storage constants
  // at the current allocations, which move whenever the cache grows.
  virtual runtime::Error rebind_for_execute(CudaDelegateHandle* handle) = 0;

  // Between steps. prepare_step() admits a step of write_length tokens and
  // grows the flat layers to fit it; commit_step() advances the logical length
  // past it. `stream` is the stream the step executes on: the growth copy and
  // the release of the old storage are ordered on it, behind every earlier
  // step that read that storage and ahead of the kernels that read the new.
  //
  // Named apart from the base's commit(SeqStepPlan), which applies one layer's
  // already-computed plan: these take a token count and speak for the whole
  // cache.
  virtual runtime::Error prepare_step(
      int64_t write_length,
      cudaStream_t stream) = 0;
  virtual runtime::Error commit_step(int64_t write_length) = 0;

  virtual OffGraphKVMetrics metrics() const = 0;
};

// Compile spec naming where a method's inputs carry the step width.
inline constexpr char kOffGraphKVStepWidthSpec[] = "offgraph_kv_step_width";

namespace detail {
// Parses a non-negative decimal int that spans the whole of `text`.
inline bool parse_index(std::string_view text, int& out) {
  if (text.empty() || text.size() > 9) {
    return false;
  }
  int value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return false;
    }
    value = value * 10 + (c - '0');
  }
  out = value;
  return true;
}
} // namespace detail

// Parses the "input_index:dim" value of kOffGraphKVStepWidthSpec. Both fields
// must be non-negative decimal integers.
inline runtime::Result<OffGraphKVStepWidth> parse_offgraph_kv_step_width(
    std::string_view value) {
  const size_t colon = value.find(':');
  OffGraphKVStepWidth where;
  ET_CHECK_OR_RETURN_ERROR(
      colon != std::string_view::npos &&
          detail::parse_index(value.substr(0, colon), where.input) &&
          detail::parse_index(value.substr(colon + 1), where.dim),
      InvalidArgument,
      "%s must be \"input_index:dim\", got '%.*s'",
      kOffGraphKVStepWidthSpec,
      static_cast<int>(value.size()),
      value.data());
  return where;
}

// The number of tokens this step writes: the extent of the named input along
// the named dim. Shape metadata only -- never the tensor's contents, which
// live on device and would cost a synchronisation every step.
inline runtime::Result<int64_t> read_offgraph_kv_step_width(
    const OffGraphKVStepWidth& where,
    runtime::Span<runtime::EValue*> inputs) {
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<size_t>(where.input) < inputs.size(),
      InvalidArgument,
      "offgraph_kv: step-width input %d is out of range (%zu inputs)",
      where.input,
      inputs.size());
  const runtime::EValue* input = inputs[where.input];
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr && input->isTensor(),
      InvalidArgument,
      "offgraph_kv: step-width input %d is not a tensor",
      where.input);
  const auto& tensor = input->toTensor();
  ET_CHECK_OR_RETURN_ERROR(
      where.dim < tensor.dim(),
      InvalidArgument,
      "offgraph_kv: step-width dim %d is out of range for a %zd-D input",
      where.dim,
      static_cast<ssize_t>(tensor.dim()));
  const int64_t width = tensor.size(where.dim);
  ET_CHECK_OR_RETURN_ERROR(
      width > 0,
      InvalidArgument,
      "offgraph_kv: step width is %lld",
      static_cast<long long>(width));
  return width;
}

// Whether the program's constants include off-graph KV storage, which only a
// runtime cache can supply. Read from the program itself, so it holds whether
// or not the caller passed a cache.
inline runtime::Result<bool> requires_offgraph_kv_storage(
    const CudaDelegateHandle& handle) {
  if (!handle.get_num_constants || !handle.get_constant_original_fqn) {
    return false;
  }
  size_t count = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      handle.get_num_constants(handle.container_handle, &count));
  for (size_t index = 0; index < count; ++index) {
    const char* fqn = nullptr;
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle.get_constant_original_fqn(handle.container_handle, index, &fqn));
    if (fqn != nullptr &&
        std::string_view(fqn).substr(0, kOffGraphKVFqnPrefix.size()) ==
            kOffGraphKVFqnPrefix) {
      return true;
    }
  }
  return false;
}

// Load time. Resolves the cache the runner published under `cache_key` and
// associates it with `handle` if the program carries off-graph storage; a
// program that does not (an embedding or vision pass) is left without one.
// A program that does needs `step_width` from its compile specs.
//
// Defined only when the backend is built with EXECUTORCH_BUILD_EXTENSION_LLM,
// whose neutral cache the off-graph cache builds on.
runtime::Error attach_offgraph_kv_cache(
    CudaDelegateHandle& handle,
    const char* cache_key,
    const std::optional<OffGraphKVStepWidth>& step_width);

// Builder for cache::kind::kSingle. Returns null when the geometry or config is
// invalid, so CacheFactory reports the failure instead of the cache asserting
// during construction.
//
// Flat layers start at cfg.initial_capacity rows and grow geometrically up to
// cfg.capacity; ring layers hold window + max_write - 1 rows from the start.
// cfg.kv_dtype is an ExecuTorch ScalarType; the slim enum the storage uses
// shares its numbering, and unsupported values are rejected here.
std::shared_ptr<cache::Cache> make_cuda_sequence_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg);

} // namespace executorch::backends::cuda
