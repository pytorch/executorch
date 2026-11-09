/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/dtype_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
// #include <cstdint>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {
/**
 * Writes the cumulative sums of a Half or BFloat16 tensor along a dim that has
 * more than one trailing position. It keeps the float sums of a block of up to
 * 1024 trailing positions and walks the rows once per block; a narrower block
 * made wide inputs much slower. The block costs 4 KB of stack, so this is not
 * inlined: callers that do not get here do not reserve it.
 */
template <typename CTYPE_OUT, typename LoadFn>
ET_NOINLINE void cumsum_in_float_blocks(
    const Tensor& self,
    LoadFn load_self,
    size_t leading_dims,
    size_t dim_size,
    size_t trailing_dims,
    CTYPE_OUT* output_data_base) {
  const char* const input_data_base =
      reinterpret_cast<const char*>(self.const_data_ptr());
  constexpr size_t kBlock = 1024;
  float sums[kBlock];
  for (size_t i = 0; i < leading_dims; i++) {
    const size_t start_loc = i * (trailing_dims * dim_size);
    for (size_t first = 0; first < trailing_dims; first += kBlock) {
      const size_t block = std::min(kBlock, trailing_dims - first);
      for (size_t j = 0; j < dim_size; j++) {
        const size_t round_base = start_loc + j * trailing_dims + first;
        for (size_t idx = 0; idx < block; idx++) {
          const float value = load_self(
              &input_data_base[(round_base + idx) * self.element_size()]);
          sums[idx] = j == 0 ? value : sums[idx] + value;
          output_data_base[round_base + idx] =
              static_cast<CTYPE_OUT>(sums[idx]);
        }
      }
    }
  }
}

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
 *
 * Half and BFloat16 cannot keep their running sums in out, because rounding
 * every partial sum to 16 bits makes long sums stop growing. They keep them in
 * float instead: one sum along the last dim, and cumsum_in_float_blocks
 * otherwise.
 */
template <typename CTYPE_OUT, typename LoadFn = CTYPE_OUT (*)(const void*)>
void cumsum_tensors(
    const Tensor& self,
    LoadFn load_self,
    int64_t dim,
    Tensor& out) {
  if (self.numel() == 0) {
    return;
  }

  const char* const input_data_base =
      reinterpret_cast<const char*>(self.const_data_ptr());
  CTYPE_OUT* output_data_base = out.mutable_data_ptr<CTYPE_OUT>();

  if (self.dim() == 0) {
    output_data_base[0] = load_self(&input_data_base[0]);
    return;
  }

  const size_t dim_size = static_cast<size_t>(self.size(dim));
  const size_t leading_dims = getLeadingDims(self, dim);
  const size_t trailing_dims = getTrailingDims(self, dim);

  using ACC = typename executorch::runtime::
      promote_types<CTYPE_OUT, CTYPE_OUT, /*half_to_float=*/true>::type;
  if constexpr (std::is_same_v<ACC, CTYPE_OUT>) {
    for (size_t i = 0; i < leading_dims; i++) {
      size_t start_loc = i * (trailing_dims * dim_size);

      for (size_t idx = 0; idx < trailing_dims; idx++) {
        output_data_base[start_loc + idx] = load_self(
            &input_data_base[(start_loc + idx) * self.element_size()]);
      }

      for (size_t j = 1; j < dim_size; j++) {
        size_t cur_round_base = start_loc + j * trailing_dims;
        size_t prev_round_base = start_loc + (j - 1) * trailing_dims;
        for (size_t idx = 0; idx < trailing_dims; idx++) {
          output_data_base[cur_round_base + idx] =
              load_self(&input_data_base
                            [(cur_round_base + idx) * self.element_size()]) +
              output_data_base[prev_round_base + idx];
        }
      }
    }
  } else if (trailing_dims == 1) {
    for (size_t i = 0; i < leading_dims; i++) {
      const size_t start_loc = i * dim_size;
      ACC sum = 0;
      for (size_t j = 0; j < dim_size; j++) {
        sum +=
            load_self(&input_data_base[(start_loc + j) * self.element_size()]);
        output_data_base[start_loc + j] = static_cast<CTYPE_OUT>(sum);
      }
    }
  } else {
    cumsum_in_float_blocks<CTYPE_OUT>(
        self,
        load_self,
        leading_dims,
        dim_size,
        trailing_dims,
        output_data_base);
  }
}

} // namespace

/**
 * Returns the cumulative sum of elements of input in the dimension dim.
 * If dtype is specified, the input tensor is casted to dtype before the
 * operation is performed. This is useful for preventing data type overflows.
 */
Tensor& cumsum_out(
    KernelRuntimeContext& ctx,
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
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, self.sizes()) == Error::Ok, InvalidArgument, out);

  dim = (self.dim() == 0) ? 0 : dim < 0 ? dim + self.dim() : dim;

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(op_name, "cumsum.out");

  ET_SWITCH_REALHBBF16_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&]() {
    const auto load_self =
        utils::internal::get_load_to_compute_fn<CTYPE_OUT, op_name>(
            ctx, self, utils::SupportedTensorDtypes::REALHBBF16);
    cumsum_tensors<CTYPE_OUT>(self, load_self, dim, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
