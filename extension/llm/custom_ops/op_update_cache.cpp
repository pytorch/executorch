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
    bool is_seq_dim_2,
    const optional<Tensor>& indices = nullopt) {
  ET_CHECK_OR_RETURN_FALSE(
      quantized_cache.dim() == 4, "quantized cache must be a 4D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      quantized_value.dim() == 4, "quantized_value must be a 4D tensor");

  // Determine the sequence dimension based on is_seq_dim_2
  int64_t seq_dim = is_seq_dim_2 ? 2 : 1;

  if (indices.has_value()) {
    const auto& indices_tensor = indices.value();
    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.dim() == 2,
        "indices must be a 2D tensor [batch_size, seq_len]");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.size(0) == quantized_value.size(0),
        "indices batch dimension must match value batch dimension");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.size(1) == quantized_value.size(seq_dim),
        "indices sequence length dimension must match value sequence length dimension");

    ET_CHECK_OR_RETURN_FALSE(
        indices_tensor.scalar_type() == ScalarType::Long,
        "indices must be of Long (int64_t) type");

    ET_CHECK_OR_RETURN_FALSE(
        is_contiguous_dim_order(
            indices_tensor.dim_order().data(), indices_tensor.dim()),
        "indices must be in contiguous dim order");
  } else {
    ET_CHECK_OR_RETURN_FALSE(
        start_pos < quantized_cache.size(seq_dim),
        "start_pos: %" PRId64 " must be less than cache size at dim %" PRId64
        ": %zd",
        start_pos,
        seq_dim,
        quantized_cache.size(seq_dim));

    ET_CHECK_OR_RETURN_FALSE(
        (start_pos + seq_length) <= quantized_cache.size(seq_dim),
        "start_post + seq_length must be less than max seq length supported by cache."
        "start pos: %" PRId64 ", seq_length: %" PRId64
        "."
        "cache size: %zd",
        start_pos,
        seq_length,
        quantized_cache.size(seq_dim));
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
    bool is_seq_dim_2,
    Tensor& output,
    const optional<Tensor>& indices = nullopt) {
  (void)ctx;

  // Determine dimensions based on is_seq_dim_2
  // If is_seq_dim_2 is false: [batch, seq, heads, head_dim]
  // If is_seq_dim_2 is true:  [batch, heads, seq, head_dim]
  int64_t value_batch_size = value.size(0);
  int64_t value_seq_len = is_seq_dim_2 ? value.size(2) : value.size(1);
  int64_t value_num_heads = is_seq_dim_2 ? value.size(1) : value.size(2);
  int64_t value_head_dim = value.size(3);

  int64_t cache_batch_size = cache.size(0);
  int64_t cache_seq_len = is_seq_dim_2 ? cache.size(2) : cache.size(1);
  int64_t cache_num_heads = is_seq_dim_2 ? cache.size(1) : cache.size(2);
  int64_t cache_head_dim = cache.size(3);

  ET_CHECK_MSG(
      value_batch_size == cache_batch_size,
      "projected_value batch size (%zd) should be equal to the cache batch size (%zd).",
      value_batch_size,
      cache_batch_size);
  ET_CHECK_MSG(
      value_num_heads == cache_num_heads,
      "projected_value number of heads (%zd) should be equal to the cache number of heads (%zd).",
      value_num_heads,
      cache_num_heads);
  ET_CHECK_MSG(
      value_head_dim == cache_head_dim,
      "projected_value embedding dimension (%zd) should be equal to the cache embedding dimension (%zd).",
      value_head_dim,
      cache_head_dim);
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
  executorch::aten::StridesType cache_seq_dim_stride =
      is_seq_dim_2 ? cache_strides[2] : cache_strides[1];
  executorch::aten::StridesType cache_head_dim_stride =
      is_seq_dim_2 ? cache_strides[1] : cache_strides[2];

  auto value_strides = value.strides();
  executorch::aten::StridesType value_batch_dim_stride = value_strides[0];
  executorch::aten::StridesType value_seq_dim_stride =
      is_seq_dim_2 ? value_strides[2] : value_strides[1];
  executorch::aten::StridesType value_head_dim_stride =
      is_seq_dim_2 ? value_strides[1] : value_strides[2];

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
        (value.numel() / (value_batch_size * value_seq_len)) *
        value.element_size();
    int64_t num_values_to_copy = value_batch_size;
    executorch::aten::StridesType value_stride = value_batch_dim_stride;
    executorch::aten::StridesType cache_stride = cache_batch_dim_stride;
    if (is_seq_dim_2) {
        /*
          If is_seq_dim_2 is true, the value tensor is in the format
          [batch, heads, seq, head_dim]. We assume we collapse this in
          [batch * heads, seq, head_dim]. Thus the stride on the first dim
          is actually stride of the heads dim
          Then for each value in [batch * heads], we copy value tensor at the index
          corresponding to the indices tensor.
          Number of bytes to copy is seqlen (value.size(seq_dim)) * head_dim * element_size
        */
        num_values_to_copy = value_batch_size * value_num_heads;
        value_stride = value_head_dim_stride;
        cache_stride = cache_head_dim_stride;
        bytes_per_token = value.size(3) * value.element_size();
    }

    for (int64_t value_idx = 0; value_idx < num_values_to_copy; ++value_idx) {
      for (int64_t seq_idx = 0; seq_idx < value_seq_len; ++seq_idx) {
        int64_t batch_index = value_idx;
        if (is_seq_dim_2) {
          batch_index = value_idx / value_num_heads;
        }
        // Get the target position from the indices tensor
        int64_t target_pos = indices_data
            [batch_index * indices_batch_stride + seq_idx * indices_seq_stride];

        // Ensure the target position is valid
        ET_CHECK_MSG(
            target_pos >= 0 && target_pos < cache_seq_len,
            "Index out of bounds: %" PRId64 " not in [0, %zd)",
            target_pos,
            cache_seq_len);

        // Calculate offsets for cache and value
        executorch::aten::SizesType cache_pos_offset =
            (value_idx * cache_stride +
             target_pos * cache_seq_dim_stride) *
            cache.element_size();

        executorch::aten::SizesType value_pos_offset =
            (value_idx * value_stride +
             seq_idx * value_seq_dim_stride) *
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
    int64_t num_values_to_copy = value_batch_size;
    executorch::aten::SizesType num_bytes_to_copy =
        (value.numel() / value_batch_size) * value.element_size();
    executorch::aten::StridesType value_stride = value_batch_dim_stride;
    executorch::aten::StridesType cache_stride = cache_batch_dim_stride;
    if (is_seq_dim_2) {
        /*
          If is_seq_dim_2 is true, the value tensor is in the format
          [batch, heads, seq, head_dim]. We assume we collapse this in
          [batch * heads, seq, head_dim]. Thus the stride on the first dim
          is actually stride of the heads dim
          Then for each value in [batch * heads], we copy value tensor at that index
          in the cache, starting at the start_pos.
          Number of bytes to copy is seqlen (value.size(seq_dim)) * head_dim * element_size
        */
        num_values_to_copy = value_batch_size * value_num_heads;
        num_bytes_to_copy = (value.numel() / (value_batch_size * value_num_heads)) * value.element_size();
        value_stride = value_head_dim_stride;
        cache_stride = cache_head_dim_stride;
    }

    for (int64_t value_idx = 0; value_idx < num_values_to_copy; ++value_idx) {
      executorch::aten::SizesType cache_pos_offset =
          (value_idx * cache_stride +
           start_pos * cache_seq_dim_stride) *
          cache.element_size();
      executorch::aten::SizesType value_pos_offset =
          (value_idx * value_stride) * cache.element_size();

      std::memcpy(
          (uint8_t*)cache_data + cache_pos_offset,
          (uint8_t*)value_data + value_pos_offset,
          num_bytes_to_copy);
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
    bool is_seq_dim_2,
    Tensor& output) {
  int64_t seq_len = is_seq_dim_2 ? value.size(2) : value.size(1);
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(value, cache, start_pos, seq_len, is_seq_dim_2),
      InvalidArgument,
      output);

  return update_cache_impl(ctx, value, cache, start_pos, is_seq_dim_2, output);
}

// New function that explicitly takes indices
Tensor& update_cache_with_indices_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    bool is_seq_dim_2,
    Tensor& output) {
  int64_t seq_len = is_seq_dim_2 ? value.size(2) : value.size(1);
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(
          value, cache, start_pos, seq_len, is_seq_dim_2, indices),
      InvalidArgument,
      output);

  return update_cache_impl(
      ctx, value, cache, start_pos, is_seq_dim_2, output, indices);
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
