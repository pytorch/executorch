/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>
#include <executorch/backends/cadence/fusion_g3/operators/xt_utils.h>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/transpose_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace G3 {
namespace native {

/**
 * Swaps dimension 'dim0' of 'a' with 'dim1', and copying
 * that mutation into `out` in a manner such that the data is densely packed
 * and is_contiguous() would return true (stride dim[size-1] = 1).
 *
 * transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out)
 */
Tensor& transpose_copy_int_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  (void)ctx;
  int kTensorDimensionLimit = 5;

  if (dim0 < 0) {
    dim0 += executorch::runtime::nonzero_dim(in);
  }
  if (dim1 < 0) {
    dim1 += executorch::runtime::nonzero_dim(in);
  }

#ifdef OP_ARG_CHECK
  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  torch::executor::get_transpose_out_target_size(
      in, dim0, dim1, expected_out_size, &expected_out_dim);

  // Resize for dynamic shape
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(
          out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);
#endif

  int inp_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];
  int permute_vec[kTensorDimensionLimit];

  /* input shapes and output shapes */
  for (int i = 0; i < in.dim(); i++) {
    inp_shape[i] = in.size(i);
  }
  for (int i = 0; i < out.dim(); i++) {
    out_shape[i] = out.size(i);
  }

  for (int i = 0; i < in.dim(); i++) {
    permute_vec[i] = i;
  }

  permute_vec[dim0] = dim1;
  permute_vec[dim1] = dim0;

  signed char* const out_data = out.mutable_data_ptr<signed char>();
  const signed char* const inp_data = in.const_data_ptr<signed char>();

  if ((in.scalar_type() == out.scalar_type()) &&
      ((out.scalar_type() == ScalarType::Int) ||
       (out.scalar_type() == ScalarType::Short) ||
       (out.scalar_type() == ScalarType::Char) ||
       (out.scalar_type() == ScalarType::UInt32) ||
       (out.scalar_type() == ScalarType::UInt16) ||
       (out.scalar_type() == ScalarType::Byte)) &&
      (in.dim() <= kTensorDimensionLimit)) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_permute,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        permute_vec,
        in.dim(),
        get_element_size(out.scalar_type()));
  } else {
    ET_KERNEL_CHECK(
        ctx,
        torch::executor::check_transpose_copy_args(in, dim0, dim1, out),
        InvalidArgument,
        out);

    ET_SWITCH_ALL_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
      torch::executor::transpose_tensors<CTYPE>(in, dim0, dim1, out);
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
