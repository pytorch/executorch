/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_update_cache.h>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace torch {
namespace executor {

namespace native {

namespace {
// Helper function to validate cache parameters
bool validate_cache_params(
    const Tensor& quantized_value,
    const Tensor& quantized_cache,
    int64_t start_pos,
    int64_t seq_length,
    const optional<Tensor>& indices = nullopt) {
  ET_CHECK_OR_RETURN_FALSE(
      quantized_cache.dim() == 4, "quantized cache must be a 4D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      quantized_value.dim() == 4, "quantized_value must be a 4D tensor");

  if (indices.has_value()) {
    const auto& indices_tensor = indices.value();
    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.dim() == 2,
        "indices must be a 2D tensor [batch_size, seq_len]");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.size(0) == quantized_value.size(0),
        "indices batch dimension must match value batch dimension");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.size(1) == quantized_value.size(1),
        "indices sequence length dimension must match value sequence length dimension");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.scalar_type() == ScalarType::Long,
        "indices must be of Long (int64_t) type");

    ET_CHECK_OR_RETURN_FALSE(
        is_contiguous_dim_order(
            indices_tensor.dim_order().data(), indices_tensor.dim()),
        "indices must be in contiguous dim order");
  } else {
    // For ring buffer support, we only check that seq_length fits in the cache
    // and that start_pos is non-negative. The actual positions will wrap around.
    ET_CHECK_OR_RETURN_FALSE(
        start_pos >= 0,
        "start_pos must be non-negative, got: %" PRId64,
        start_pos);

    ET_CHECK_OR_RETURN_FALSE(
        seq_length <= quantized_cache.size(1),
        "seq_length (%" PRId64 ") must be <= cache size (%zd)",
        seq_length,
        quantized_cache.size(1));
  }

  // Make sure they are in contiguous dim order
  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(
          quantized_cache.dim_order().data(), quantized_cache.dim()),
      "quantized cache must be in contiguous dim order");

  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(
          quantized_value.dim_order().data(), quantized_value.dim()),
      "quantized value must be in contiguous dim order");

  return true;
}

// Helper function for the actual update operation
Tensor& update_cache_impl(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output,
    const optional<Tensor>& indices = nullopt) {
  (void)ctx;

  ET_CHECK_MSG(
      value.size(0) == cache.size(0),
      "projected_value batch size (%zd) should be equal to the cache batch size (%zd).",
      value.size(0),
      cache.size(0));
  ET_CHECK_MSG(
      value.size(2) == cache.size(2),
      "projected_value number of heads (%zd) should be equal to the cache number of heads (%zd).",
      value.size(2),
      cache.size(2));
  ET_CHECK_MSG(
      value.size(3) == cache.size(3),
      "projected_value embedding dimension (%zd) should be equal to the cache embedding dimension (%zd).",
      value.size(3),
      cache.size(3));
  ET_CHECK_MSG(
      value.element_size() == cache.element_size(),
      "projected_value data type size (%zd) should be equal to the cache data type size (%zd).",
      value.element_size(),
      cache.element_size());

  ET_CHECK_MSG(
      is_contiguous_dim_order(value.dim_order().data(), value.dim()),
      "projected value must be in contiguous dim order");
  ET_CHECK_MSG(
      is_contiguous_dim_order(cache.dim_order().data(), cache.dim()),
      "projected value must be in contiguous dim order");

  const void* value_data = value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();

  ET_CHECK_MSG(value_data, "projected_value data is null");
  ET_CHECK_MSG(cache_data, "cache data is null");

  auto cache_strides = cache.strides();
  executorch::aten::StridesType cache_batch_dim_stride = cache_strides[0];
  executorch::aten::StridesType cache_seq_dim_stride = cache_strides[1];

  auto value_strides = value.strides();
  executorch::aten::StridesType value_batch_dim_stride = value_strides[0];

  executorch::aten::SizesType num_bytes_to_copy =
      (value.numel() / value.size(0)) * value.element_size();

  if (indices.has_value()) {
    // Use the provided indices tensor for each batch and sequence position
    const Tensor& indices_tensor = indices.value();
    const int64_t* indices_data =
        static_cast<const int64_t*>(indices_tensor.const_data_ptr());
    auto indices_strides = indices_tensor.strides();
    executorch::aten::StridesType indices_batch_stride = indices_strides[0];
    executorch::aten::StridesType indices_seq_stride = indices_strides[1];

    // Calculate bytes to copy for a single token
    executorch::aten::SizesType bytes_per_token =
        (value.numel() / (value.size(0) * value.size(1))) *
        value.element_size();

    for (int64_t batch_line = 0; batch_line < value.size(0); ++batch_line) {
      for (int64_t seq_idx = 0; seq_idx < value.size(1); ++seq_idx) {
        // Get the target position from the indices tensor
        int64_t target_pos = indices_data
            [batch_line * indices_batch_stride + seq_idx * indices_seq_stride];

        // Ensure the target position is valid
        ET_CHECK_MSG(
            target_pos >= 0 && target_pos < cache.size(1),
            "Index out of bounds: %" PRId64 " not in [0, %zd)",
            target_pos,
            cache.size(1));

        // Calculate offsets for cache and value
        executorch::aten::SizesType cache_pos_offset =
            (batch_line * cache_batch_dim_stride +
             target_pos * cache_seq_dim_stride) *
            cache.element_size();

        executorch::aten::SizesType value_pos_offset =
            (batch_line * value_batch_dim_stride + seq_idx * value_strides[1]) *
            value.element_size();

        // Copy a single token
        std::memcpy(
            (uint8_t*)cache_data + cache_pos_offset,
            (uint8_t*)value_data + value_pos_offset,
            bytes_per_token);
      }
    }
  } else {
    // Use the original implementation with start_pos
    // Support ring buffer by wrapping positions if they exceed cache size
    int64_t cache_seq_len = cache.size(1);
    int64_t value_seq_len = value.size(1);

    for (int64_t batch_line = 0; batch_line < value.size(0); ++batch_line) {
      // Check if we need to handle wrapping
      if (start_pos + value_seq_len <= cache_seq_len) {
        // No wrapping needed - single contiguous copy
        executorch::aten::SizesType cache_pos_offset =
            (batch_line * cache_batch_dim_stride +
             start_pos * cache_seq_dim_stride) *
            cache.element_size();
        executorch::aten::SizesType value_pos_offset =
            (batch_line * value_batch_dim_stride) * cache.element_size();

        std::memcpy(
            (uint8_t*)cache_data + cache_pos_offset,
            (uint8_t*)value_data + value_pos_offset,
            num_bytes_to_copy);
      } else {
        // Ring buffer wrapping needed - copy in two parts
        // Part 1: from start_pos to end of cache
        int64_t first_part_len = cache_seq_len - start_pos;
        // Part 2: from beginning of cache (wrapped around)
        int64_t second_part_len = value_seq_len - first_part_len;

        executorch::aten::SizesType bytes_per_token =
            (value.numel() / (value.size(0) * value.size(1))) *
            value.element_size();

        // Copy first part (start_pos to end of cache)
        if (first_part_len > 0) {
          executorch::aten::SizesType cache_pos_offset =
              (batch_line * cache_batch_dim_stride +
               start_pos * cache_seq_dim_stride) *
              cache.element_size();
          executorch::aten::SizesType value_pos_offset =
              (batch_line * value_batch_dim_stride) * cache.element_size();

          std::memcpy(
              (uint8_t*)cache_data + cache_pos_offset,
              (uint8_t*)value_data + value_pos_offset,
              first_part_len * bytes_per_token);
        }

        // Copy second part (beginning of cache, wrapped)
        if (second_part_len > 0) {
          executorch::aten::SizesType cache_pos_offset =
              (batch_line * cache_batch_dim_stride) * cache.element_size();
          executorch::aten::SizesType value_pos_offset =
              (batch_line * value_batch_dim_stride +
               first_part_len * value_strides[1]) *
              value.element_size();

          std::memcpy(
              (uint8_t*)cache_data + cache_pos_offset,
              (uint8_t*)value_data + value_pos_offset,
              second_part_len * bytes_per_token);
        }
      }
    }
  }

  // Noone uses output. Just a placeholder.
  return output;
}
} // anonymous namespace

// Original update_cache_out function without indices parameter
Tensor& update_cache_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output) {
  int64_t seq_len = value.size(1);
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(value, cache, start_pos, seq_len),
      InvalidArgument,
      output);

  return update_cache_impl(ctx, value, cache, start_pos, output);
}

// New function that explicitly takes indices
Tensor& update_cache_with_indices_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    Tensor& output) {
  int64_t seq_len = value.size(1);
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(value, cache, start_pos, seq_len, indices),
      InvalidArgument,
      output);

  return update_cache_impl(ctx, value, cache, start_pos, output, indices);
}

} // namespace native
} // namespace executor
} // namespace torch

// Really this is just an inplace tensor update op
// which makes assumption on the rank of a tensor,
// and the dim order (memory layout) of the tensor.
// Furthermore assumes that the indexing is along
// sequence dimension (dim 1) of the tensor.
// In later diffs will rename this to update_cache.
EXECUTORCH_LIBRARY(
    llama,
    "update_cache.out",
    torch::executor::native::update_cache_out);

// Register the new update_cache_with_indices.out op
EXECUTORCH_LIBRARY(
    llama,
    "update_cache_with_indices.out",
    torch::executor::native::update_cache_with_indices_out);
