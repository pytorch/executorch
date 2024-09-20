/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_update_quantized_cache.h>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <array>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <vector>

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

Tensor& update_quantized_cache_out(
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
  ET_CHECK_MSG(value.dim() == 4, "value must be a 4D tensor");

  ET_CHECK_MSG(value.size(0) == 1, "value must have batch size of 1");
  ET_CHECK_MSG(cache.size(0) == 1, "cache must have batch size of 1");
  const void* value_data = value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();

  ET_CHECK_MSG(value_data != nullptr, "projected_value data is null");
  ET_CHECK_MSG(cache_data, "cache data is null");

  auto strides = cache.strides();
  exec_aten::StridesType seq_dim_stride = strides[1];
  exec_aten::SizesType pos_offset = start_pos * seq_dim_stride;
  exec_aten::SizesType pos_offset_bytes = pos_offset * value.element_size();
  exec_aten::SizesType num_bytes = value.numel() * value.element_size();
  // NOLINTNEXTLINE
  std::memcpy((uint8_t*)cache_data + pos_offset_bytes, value_data, num_bytes);

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
    "update_quantized_cache.out",
    torch::executor::native::update_quantized_cache_out);
