/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_causal_mask_gen.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {
namespace {

template <typename Position>
bool positions_are_valid(const Position* positions, int64_t num_positions) {
  for (int64_t row = 0; row < num_positions; ++row) {
    if (positions[row] < 0) {
      return false;
    }
  }
  return true;
}

template <typename Position>
void generate_mask(
    const Position* positions,
    int64_t num_positions,
    int64_t packed_width,
    uint8_t* out) {
  for (int64_t row = 0; row < num_positions; ++row) {
    uint8_t* row_out = out + row * packed_width;
    const int64_t visible = positions[row] >= packed_width * 8
        ? packed_width * 8
        : static_cast<int64_t>(positions[row]) + 1;
    const int64_t full_visible_bytes = visible / 8;
    const int64_t partial_visible_bits = visible % 8;

    std::memset(row_out, 0, static_cast<std::size_t>(full_visible_bytes));
    int64_t next_byte = full_visible_bytes;
    if (partial_visible_bits != 0) {
      row_out[next_byte++] =
          static_cast<uint8_t>(0xFFu << partial_visible_bits);
    }
    std::memset(
        row_out + next_byte,
        0xFF,
        static_cast<std::size_t>(packed_width - next_byte));
  }
}

} // namespace

::executorch::aten::Tensor& sdpa_bitwise_causal_mask_gen_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& positions,
    int64_t key_length,
    ::executorch::aten::Tensor& out) {
  const auto position_sizes = positions.sizes();
  ET_KERNEL_CHECK(ctx, position_sizes.size() == 1, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, key_length > 0 && key_length % 8 == 0, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      positions.dtype() == ::executorch::aten::ScalarType::Int ||
          positions.dtype() == ::executorch::aten::ScalarType::Long,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx,
      out.dtype() == ::executorch::aten::ScalarType::Byte,
      InvalidArgument,
      out);

  const int64_t num_positions = positions.numel();
  const int64_t packed_width = key_length / 8;
  ET_KERNEL_CHECK(
      ctx,
      out.dim() == 2 && out.size(0) == num_positions &&
          out.size(1) == packed_width,
      InvalidArgument,
      out);

  const bool valid_positions =
      positions.dtype() == ::executorch::aten::ScalarType::Long
      ? positions_are_valid(positions.const_data_ptr<int64_t>(), num_positions)
      : positions_are_valid(positions.const_data_ptr<int32_t>(), num_positions);
  ET_KERNEL_CHECK(ctx, valid_positions, InvalidArgument, out);

  if (positions.dtype() == ::executorch::aten::ScalarType::Long) {
    generate_mask(
        positions.const_data_ptr<int64_t>(),
        num_positions,
        packed_width,
        out.mutable_data_ptr<uint8_t>());
  } else {
    generate_mask(
        positions.const_data_ptr<int32_t>(),
        num_positions,
        packed_width,
        out.mutable_data_ptr<uint8_t>());
  }
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
