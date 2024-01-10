/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>
#include <cstddef>
//#include <cstdint>
//#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {
/**
 * Returns the cumulative sum of elements of input in the dimension dim.
 *
 * Given a self tensor whose size is (d1, d2, .., d_dim, ..., dm), and does
 * cumsum along dim, we first copy all values in self[d1, d2, .., 0, ..., dm]
 * to out[d1, d2, .., 0, ..., dm] since no cumsum should be done for the
 * first element. Then calculate all out[d1, d2, .., i, ..., dm] by adding
 * out[d1, d2, .., i-1, ..., dm] and self[d1, d2, .., i-1, ..., dm].
 * This approach ensures that computations are sequential rather than jumpy at
 * the memory level, thereby increasing the speed of memory IO as
 * well as reducing the number of cache misses.
 */
template <typename CTYPE_IN, typename CTYPE_OUT>
void cumsum_tensors(const Tensor& self, int64_t dim, Tensor& out) {
  if (self.numel() == 0) {
    return;
  }

  const CTYPE_IN* input_data_base = self.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT* output_data_base = out.mutable_data_ptr<CTYPE_OUT>();

  if (self.dim() == 0) {
    output_data_base[0] = input_data_base[0];
    return;
  }

  const size_t dim_size = static_cast<size_t>(self.size(dim));
  const size_t leading_dims = getLeadingDims(self, dim);
  const size_t trailing_dims = getTrailingDims(self, dim);

  for (size_t i = 0; i < leading_dims; i++) {
    size_t start_loc = i * (trailing_dims * dim_size);

    for (size_t idx = 0; idx < trailing_dims; idx++) {
      output_data_base[start_loc + idx] =
          static_cast<CTYPE_OUT>(input_data_base[start_loc + idx]);
    }

    for (size_t j = 1; j < dim_size; j++) {
      size_t cur_round_base = start_loc + j * trailing_dims;
      size_t prev_round_base = start_loc + (j - 1) * trailing_dims;
      for (size_t idx = 0; idx < trailing_dims; idx++) {
        output_data_base[cur_round_base + idx] =
            static_cast<CTYPE_OUT>(input_data_base[cur_round_base + idx]) +
            output_data_base[prev_round_base + idx];
      }
    }
  }
}

} // namespace

/**
 * Returns the cumulative sum of elements of input in the dimension dim.
 * If dtype is specified, the input tensor is casted to dtype before the
 * operation is performed. This is useful for preventing data type overflows.
 */
Tensor& cumsum_out(
    RuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    optional<ScalarType> enforced_dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_cumsum_args(self, dim, enforced_dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, self.sizes()) == Error::Ok, InvalidArgument, out);

  dim = (self.dim() == 0) ? 0 : dim < 0 ? dim + self.dim() : dim;

// Use a two layer switch to handle each possible data pair
#define CUMSUM_IMPL(SELF_CTYPE, OUT_CTYPE, out_dtype)      \
  case ScalarType::out_dtype:                              \
    cumsum_tensors<SELF_CTYPE, OUT_CTYPE>(self, dim, out); \
    break;

#define CUMSUM_TENSORS(SELF_CTYPE, self_dtype)           \
  case ScalarType::self_dtype:                           \
    switch (out.scalar_type()) {                         \
      ET_FORALL_REAL_TYPES_WITH(SELF_CTYPE, CUMSUM_IMPL) \
      default:                                           \
        ET_CHECK_MSG(                                    \
            false,                                       \
            "Unhandled output dtype %" PRId8,            \
            static_cast<int8_t>(out.scalar_type()));     \
    }                                                    \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, CUMSUM_TENSORS)
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(self.scalar_type()));
  }

#undef CUMSUM_TENSORS
#undef CUMSUM_IMPL

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
