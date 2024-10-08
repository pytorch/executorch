/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Clears `out` by setting all elements to 0.
 */
Tensor& clear_out(Tensor& out) {
  uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
  if (out_data != nullptr) {
    memset(out_data, 0, out.nbytes());
  }
  return out;
}

/**
 * Applies lower-triangular part of `self` to `out` using parameters defined.
 * This function is agnostic to whether `self` is a 2D matrix or batch of
 * matrices.
 */
template <typename CTYPE>
void apply_tril(
    CTYPE* __restrict__ self,
    CTYPE* __restrict__ out,
    int64_t diagonal,
    int64_t num_rows,
    int64_t num_cols,
    int64_t row_stride,
    int64_t col_stride) {
  for (int64_t i = 0; i < num_rows; i++) {
    for (int64_t j = 0; j < std::min(num_cols, i + diagonal + 1); j++) {
      out[i * row_stride + j * col_stride] =
          self[i * row_stride + j * col_stride];
    }
  }
}

/**
 * `tril_out` helper function.
 */
template <typename CTYPE>
void tril_kernel(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t diagonal,
    const Tensor& out) {
  // Dynamically compute `self` sizes and strides.

  int64_t ndim = self.dim();

  ET_KERNEL_CHECK_MSG(
      ctx,
      ndim < kTensorDimensionLimit,
      InvalidArgument,
      ,
      "ndim %" PRId64 " >= %zu",
      ndim,
      kTensorDimensionLimit);

  int64_t sizes[kTensorDimensionLimit];
  int64_t strides[kTensorDimensionLimit];

  for (size_t i = 0; i < ndim; ++i) {
    sizes[i] = self.size(i);
    strides[i] = getTrailingDims(self, static_cast<int64_t>(i));
  }

  IntArrayRef sizes_ref(sizes, ndim);
  IntArrayRef strides_ref(strides, ndim);

  int64_t num_rows = sizes_ref[ndim - 2];
  int64_t num_cols = sizes_ref[ndim - 1];

  // Compute `tril` for a 2D matrix or a batch of matrices. For a batch of
  // matrices, `batch_size` will be >1, and `apply_tril` will be executed
  // multiple times, each referencing a multiple of `self_stride`.

  int64_t batch_size = getLeadingDims(self, ndim - 2);
  int64_t self_stride =
      (self.dim() > 2 && strides_ref[ndim - 3] > 0) ? strides_ref[ndim - 3] : 1;

  auto data_self = self.mutable_data_ptr<CTYPE>();
  auto data_out = out.mutable_data_ptr<CTYPE>();

  int64_t row_stride = strides_ref[ndim - 2];
  int64_t col_stride = strides_ref[ndim - 1];

  for (int64_t i = 0; i < batch_size; i++) {
    CTYPE* __restrict__ data_self_ptr = &data_self[i * self_stride];
    CTYPE* __restrict__ data_out_ptr = &data_out[i * self_stride];

    apply_tril<CTYPE>(
        data_self_ptr,
        data_out_ptr,
        diagonal,
        num_rows,
        num_cols,
        row_stride,
        col_stride);
  }
}

} // namespace

/**
 * `tril_out` implementation for all dtypes (real + bool). Returns the
 * lower-triangular part of a 2D matrix or batch of matrices in `out`, where all
 * other elements are set to 0, by default. Further, `diagonal` controls how the
 * lower-triangular subset is defined:
 *    1. `diagonal = 0`: Elements on and below the main diagonal are retained.
 *    2. `diagonal > 0`: Similar to case (1); additional diagonals above the
 *       main one are also captured.
 *    3. `diagonal < 0`: Similar to case (1); additional diagonals below the
 *       main one are also captured.
 */
Tensor& tril_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t diagonal,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, check_tril_args(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(self), InvalidArgument, out);

  if (self.numel() == 0) {
    return out;
  }

  // Fill `out` with 0s prior to executing tril.
  clear_out(out);

  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, __func__, CTYPE, [&]() {
    tril_kernel<CTYPE>(ctx, self, diagonal, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
