/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

template <typename T>
void randomized_fast_hadamard_transform_impl(
    const T* vec,
    T* out,
    const std::uint8_t* s,
    int vec_size) {
  if (vec_size == 0) {
    return;
  }

  std::memcpy(out, vec, vec_size * sizeof(T));

  int step = 1;
  while (step < vec_size) {
    for (int ii = 0; ii < vec_size; ii += step * 2) {
      for (int jj = ii; jj < ii + step; ++jj) {
        auto x = out[jj];
        auto y = out[jj + step];
        out[jj] = x + y;
        out[jj + step] = x - y;
      }
    }
    step *= 2;
  }

  // vec_size is a power of 2 so an optimized implementation could use
  // 1) a lookup table and 2) potentially a faster instruction than
  // multiply if vec_size is a square.
  const T inv_sqrt = T(1) / std::sqrt(T(vec_size));
  for (int ii = 0; ii < vec_size; ++ii) {
    T adjusted_val = out[ii] * inv_sqrt;
    // This conditional negation implements matrix multiplication by
    // diag(s).
    if ((s[ii / 8] & (1 << (ii % 8))) != 0) {
      adjusted_val = -adjusted_val;
    }
    out[ii] = adjusted_val;
  }
}

template void randomized_fast_hadamard_transform_impl<float>(
    const float* vec,
    float* out,
    const std::uint8_t* randomization_bitvec,
    int vec_size);

Tensor& randomized_fast_hadamard_transform_out(
    RuntimeContext& ctx,
    const Tensor& vec,
    const Tensor& randomization_bitvec,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, vec.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, vec.scalar_type() == out.scalar_type(), InvalidArgument, out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      randomization_bitvec.scalar_type() == exec_aten::ScalarType::Byte,
      InvalidArgument,
      out,
      "randomization_bitvec argument to randomized_fast_hadamard_transform_out must be a "
      "bitvector stored as type Byte!");

  ET_KERNEL_CHECK_MSG(
      ctx,
      vec.dim() == 1,
      InvalidArgument,
      out,
      "The Hadamard transform expects a vector!");

  const auto vec_numel = vec.numel();
  ET_KERNEL_CHECK_MSG(
      ctx,
      (vec_numel & (vec_numel - 1)) == 0,
      InvalidArgument,
      out,
      "This implementation requires power-of-2 input size!");
  ET_SWITCH_FLOATH_TYPES(vec.scalar_type(), ctx, __func__, CTYPE, [&] {
    randomized_fast_hadamard_transform_impl(
        vec.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        randomization_bitvec.const_data_ptr<std::uint8_t>(),
        vec.numel());
  });
  return out;
}
} // namespace native
} // namespace executor
} // namespace torch
