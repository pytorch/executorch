/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_ring_mask_gen.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {
namespace {

void clear_range(uint8_t* out, int64_t start, int64_t end) {
  if (start == end) {
    return;
  }
  const int64_t first_byte = start / 8;
  const int64_t last_byte = (end - 1) / 8;
  if (first_byte == last_byte) {
    const uint32_t visible = ((1u << (end - start)) - 1) << (start % 8);
    out[first_byte] &= static_cast<uint8_t>(~visible);
    return;
  }
  out[first_byte] &= static_cast<uint8_t>((1u << (start % 8)) - 1);
  std::memset(
      out + first_byte + 1,
      0,
      static_cast<std::size_t>(last_byte - first_byte - 1));
  out[last_byte] &= static_cast<uint8_t>(0xFFu << ((end - 1) % 8 + 1));
}

template <typename Position>
bool generate_mask(
    const Position* positions,
    int64_t num_positions,
    int64_t sliding_window,
    int64_t cache_length,
    uint8_t* out) {
  for (int64_t row = 0; row < num_positions; ++row) {
    if (positions[row] < 0) {
      return false;
    }
  }
  const int64_t packed_width = cache_length / 8;
  for (int64_t row = 0; row < num_positions; ++row) {
    uint8_t* row_out = out + row * packed_width;
    std::memset(row_out, 0xFF, static_cast<std::size_t>(packed_width));
    const int64_t position = positions[row];
    const int64_t visible =
        position >= sliding_window - 1 ? sliding_window : position + 1;
    const int64_t end = position % cache_length + 1;
    const int64_t start = end - visible;
    if (start < 0) {
      clear_range(row_out, cache_length + start, cache_length);
      clear_range(row_out, 0, end);
    } else {
      clear_range(row_out, start, end);
    }
  }
  return true;
}

} // namespace

::executorch::aten::Tensor& sdpa_bitwise_ring_mask_gen_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& positions,
    int64_t sliding_window,
    int64_t cache_length,
    ::executorch::aten::Tensor& out) {
  using ::executorch::aten::ScalarType;
  ET_KERNEL_CHECK(ctx, positions.dim() == 1, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, cache_length > 0 && cache_length % 8 == 0, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      sliding_window > 0 && sliding_window <= cache_length,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      positions.dtype() == ScalarType::Int ||
          positions.dtype() == ScalarType::Long,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(ctx, out.dtype() == ScalarType::Byte, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      out.dim() == 2 && out.size(0) == positions.numel() &&
          out.size(1) == cache_length / 8,
      InvalidArgument,
      out);

  const bool valid = positions.dtype() == ScalarType::Long
      ? generate_mask(
            positions.const_data_ptr<int64_t>(),
            positions.numel(),
            sliding_window,
            cache_length,
            out.mutable_data_ptr<uint8_t>())
      : generate_mask(
            positions.const_data_ptr<int32_t>(),
            positions.numel(),
            sliding_window,
            cache_length,
            out.mutable_data_ptr<uint8_t>());
  ET_KERNEL_CHECK(ctx, valid, InvalidArgument, out);
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
