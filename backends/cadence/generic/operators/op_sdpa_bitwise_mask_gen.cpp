/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_mask_gen.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace impl {
namespace generic {
namespace native {

::executorch::aten::Tensor& sdpa_bitwise_mask_gen_out(
    ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& mask,
    double threshold,
    ::executorch::aten::Tensor& out) {
  // NOTE: Mask semantics and bit polarity
  //
  // PyTorch SDPA (scaled dot-product attention) commonly uses a mask where:
  //   - True (or large positive values after conversion) means "keep/allow"
  //   - False (or large negative values) means "mask/block"
  //
  // In this backend we deliberately flip the sense of the mask bit when
  // packing:
  //   - We generate a bit = 1 to indicate "masked/blocked"
  //   - We generate a bit = 0 to indicate "valid/kept"
  //
  // Rationale:
  //   - The downstream kernels expect a compact bitmask where 1 means "do not
  //   attend".
  //   - Therefore, for a boolean mask we invert the value (!value):
  //       input True  (keep)  -> bit 0
  //       input False (block) -> bit 1
  //   - For a floating mask, values strictly less than the threshold are
  //   considered "masked":
  //       input < threshold -> bit 1 (masked)
  //       else              -> bit 0 (kept)
  //
  // Packing layout:
  //   - Elements are consumed in groups of 8 along the last dimension.
  //   - The j-th element in a group maps to bit j of the output byte
  //   (LSB-first).
  //   - This produces a compact per-8-elements representation matching the
  //   kernel’s expectation.

  // Ensure the last dimension of mask is divisible by 8
  auto sizes = mask.sizes();
  ET_KERNEL_CHECK(ctx, sizes.size() > 0, InvalidArgument, out);
  auto last_dim = sizes.back();
  ET_KERNEL_CHECK(ctx, last_dim % 8 == 0, InvalidArgument, out);
  const auto dtype = mask.dtype();
  const int64_t numel = mask.numel();
  if (dtype == ::executorch::aten::ScalarType::Bool) {
    // Generate bitwise mask by iterating boolean tensor elements and inverting
    // each
    for (int64_t i = 0, out_index = 0; i < numel; i += 8, out_index++) {
      uint8_t packed_mask = 0;
      for (int64_t j = 0; j < 8; j++) {
        packed_mask |= (!mask.mutable_data_ptr<bool>()[i + j]) << j;
      }
      out.mutable_data_ptr<uint8_t>()[out_index] = packed_mask;
    }
  } else if (dtype == ::executorch::aten::ScalarType::Float) {
    for (int64_t i = 0, out_index = 0; i < numel; i += 8, out_index++) {
      uint8_t packed_mask = 0;
      for (int64_t j = 0; j < 8; j++) {
        packed_mask |= (mask.mutable_data_ptr<float>()[i + j] < threshold) << j;
      }
      out.mutable_data_ptr<uint8_t>()[out_index] = packed_mask;
    }
  } else {
    ET_KERNEL_CHECK(ctx, false, InvalidArgument, out);
  }
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
