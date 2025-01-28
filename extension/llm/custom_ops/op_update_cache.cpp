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
bool validate_cache_params(
    const Tensor& quantized_value,
    const Tensor& quantized_cache,
    int64_t start_pos,
    int64_t seq_length) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      quantized_cache.dim() == 4, "quantized cache must be a 4D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      quantized_value.dim() == 4, "quantized_value must be a 4D tensor");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      start_pos < quantized_cache.size(1),
      "start_pos must be less than cache size at dim 1");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (start_pos + seq_length) <= quantized_cache.size(1),
      "start_post + seq_length must be less than max seq length supported by cache."
      "start pos: %" PRId64 ", seq_length: %" PRId64
      "."
      "cache size: %zd",
      start_pos,
      seq_length,
      quantized_cache.size(1));

  // Make sure they are in contiguous dim order
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(
          quantized_cache.dim_order().data(), quantized_cache.dim()),
      "quantized cache must be in contiguous dim order");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      is_contiguous_dim_order(
          quantized_value.dim_order().data(), quantized_value.dim()),
      "quantized value must be in contiguous dim order");

  return true;
}
} // anonymous namespace

Tensor& update_cache_out(
    RuntimeContext& ctx,
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output) {
  (void)ctx;
  int64_t seq_len = value.size(1);
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(value, cache, start_pos, seq_len),
      InvalidArgument,
      output);

  ET_CHECK_MSG(
      value.size(0) == cache.size(0),
      "projected_value batch size should be equal to the cache batch size.");
  ET_CHECK_MSG(
      value.size(2) == cache.size(2),
      "projected_value number of heads should be equal to the cache number of heads.");
  ET_CHECK_MSG(
      value.size(3) == cache.size(3),
      "projected_value embedding dimension should be equal to the cache embedding dimension.");
  ET_CHECK_MSG(
      value.element_size() == cache.element_size(),
      "projected_value data type size should be equal to the cache data type size.");

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
  exec_aten::StridesType cache_batch_dim_stride = cache_strides[0];
  exec_aten::StridesType cache_seq_dim_stride = cache_strides[1];

  auto value_strides = value.strides();
  exec_aten::StridesType value_batch_dim_stride = value_strides[0];

  exec_aten::SizesType num_bytes_to_copy =
      (value.numel() / value.size(0)) * value.element_size();

  for (int64_t batch_line = 0; batch_line < value.size(0); ++batch_line) {
    exec_aten::SizesType cache_pos_offset =
        (batch_line * cache_batch_dim_stride +
         start_pos * cache_seq_dim_stride) *
        cache.element_size();
    exec_aten::SizesType value_pos_offset =
        (batch_line * value_batch_dim_stride) * cache.element_size();

    std::memcpy(
        (uint8_t*)cache_data + cache_pos_offset,
        (uint8_t*)value_data + value_pos_offset,
        num_bytes_to_copy);
  }

  // Noone uses output. Just a placeholder.
  return output;
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
