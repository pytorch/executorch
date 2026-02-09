/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

namespace {
Tensor& update_cross_attn_cache_out_impl(
    const Tensor& value,
    Tensor& cache,
    Tensor& out) {
  ET_CHECK_MSG(
      value.scalar_type() == cache.scalar_type(),
      "value dtype (%hhd) should be equal to cache dtype (%hhd).",
      value.scalar_type(),
      cache.scalar_type());
  ET_CHECK_MSG(
      cache.scalar_type() == out.scalar_type(),
      "cache dtype (%hhd) should be equal to out dtype (%hhd).",
      cache.scalar_type(),
      out.scalar_type());
  ET_CHECK_MSG(
      value.dim() == 4,
      "value must be 4D, got %zd",
      value.dim());
  ET_CHECK_MSG(
      cache.dim() == 4,
      "cache must be 4D, got %zd",
      cache.dim());
  ET_CHECK_MSG(
      out.dim() == 4,
      "out must be 4D, got %zd",
      out.dim());
  ET_CHECK_MSG(
      value.size(0) == cache.size(0),
      "value batch size (%zd) should be equal to cache batch size (%zd).",
      value.size(0),
      cache.size(0));
  ET_CHECK_MSG(
      value.size(1) == cache.size(1),
      "value num heads (%zd) should be equal to cache num heads (%zd).",
      value.size(1),
      cache.size(1));
  ET_CHECK_MSG(
      value.size(3) == cache.size(3),
      "value head dim (%zd) should be equal to cache head dim (%zd).",
      value.size(3),
      cache.size(3));
  ET_CHECK_MSG(
      value.size(2) <= cache.size(2),
      "value sequence length (%zd) must be <= cache sequence length (%zd).",
      value.size(2),
      cache.size(2));
  ET_CHECK_MSG(
      cache.sizes() == out.sizes(),
      "cache and out should have identical sizes.");
  ET_CHECK_MSG(
      is_contiguous_dim_order(value.dim_order().data(), value.dim()),
      "value must have contiguous dim order.");
  ET_CHECK_MSG(
      is_contiguous_dim_order(cache.dim_order().data(), cache.dim()),
      "cache must have contiguous dim order.");
  ET_CHECK_MSG(
      is_contiguous_dim_order(out.dim_order().data(), out.dim()),
      "out must have contiguous dim order.");

  const void* value_data = value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();
  void* out_data = out.mutable_data_ptr();
  ET_CHECK_MSG(
      (value_data != nullptr) || value.nbytes() == 0,
      "value data is null.");
  ET_CHECK_MSG(
      (cache_data != nullptr) || cache.nbytes() == 0,
      "cache data is null.");
  ET_CHECK_MSG(
      (out_data != nullptr) || out.nbytes() == 0,
      "out data is null.");

  // Update cache in place by writing value into prefix [0:S].
  // This mirrors update_cache behavior and keeps cross-attn cache persistent
  // across decode steps even when `out` does not alias `cache`.

  const size_t B = value.size(0);
  const size_t H = value.size(1);
  const size_t S = value.size(2);
  const size_t D = value.size(3);
  const size_t S_max = cache.size(2);

  // Calculate sizes
  const size_t elem_size = value.element_size();
  const size_t row_size = D * elem_size;
  const size_t prefix_bytes = S * row_size;
  const size_t value_batch_stride = H * S * row_size;
  const size_t value_head_stride = S * row_size;
  const size_t cache_batch_stride = H * S_max * row_size;
  const size_t cache_head_stride = S_max * row_size;
  const size_t out_batch_stride = H * S_max * row_size;
  const size_t out_head_stride = S_max * row_size;
  const bool out_aliases_cache = out_data == cache_data;

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      const char* value_ptr = static_cast<const char*>(value_data) +
          b * value_batch_stride + h * value_head_stride;
      char* cache_ptr = static_cast<char*>(cache_data) +
          b * cache_batch_stride + h * cache_head_stride;
      char* out_ptr = static_cast<char*>(out_data) +
          b * out_batch_stride + h * out_head_stride;

      // Always persist cross-attention cache update.
      if (prefix_bytes > 0) {
        std::memcpy(cache_ptr, value_ptr, prefix_bytes);
      }

      // Materialize out from the updated cache when out does not alias cache.
      if (!out_aliases_cache && cache_head_stride > 0) {
        std::memcpy(out_ptr, cache_ptr, cache_head_stride);
      }
    }
  }

  return out;
}
} // namespace

} // namespace native
} // namespace executor
} // namespace torch

namespace {
void update_cross_attn_cache_out_call(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack) {
  (void)ctx;
  // Stack layout: value, cache, out
  torch::executor::Tensor value = stack[0]->toTensor();
  torch::executor::Tensor& cache = stack[1]->toTensor();
  torch::executor::Tensor& out = stack[2]->toTensor();
  torch::executor::native::update_cross_attn_cache_out_impl(value, cache, out);
}
} // namespace

static auto update_cross_attn_cache_registration =
    ::executorch::runtime::register_kernel(::executorch::runtime::Kernel(
        "executorch::update_cross_attn_cache.out",
        update_cross_attn_cache_out_call));
