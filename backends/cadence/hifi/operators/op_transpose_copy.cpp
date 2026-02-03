/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/transpose_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::nonzero_dim;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::check_transpose_copy_args;
using torch::executor::get_transpose_out_target_size;
using torch::executor::transpose_tensors;

namespace impl {
namespace HiFi {
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

  if (dim0 < 0) {
    dim0 += nonzero_dim(in);
  }
  if (dim1 < 0) {
    dim1 += nonzero_dim(in);
  }

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_transpose_out_target_size(
      in, dim0, dim1, expected_out_size, &expected_out_dim);

  // Resize for dynamic shape
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  const auto in_type = in.scalar_type();
  constexpr int kNnlibMaxDim = 5;

  bool optimized = false;

  if (out.scalar_type() == ScalarType::Float ||
      out.scalar_type() == ScalarType::Char ||
      out.scalar_type() == ScalarType::Byte)
    optimized = true;

  if (in.dim() > kNnlibMaxDim)
    optimized = false;

  if (optimized) {
    WORD32 num_inp_dims = in.dim();
    WORD32 num_out_dims = num_inp_dims;

    WORD32 p_inp_shape[kNnlibMaxDim];
    WORD32 p_out_shape[kNnlibMaxDim];
    WORD32 p_permute_vec[kNnlibMaxDim];

    for (int i = 0; i < num_inp_dims; i++) {
      p_inp_shape[i] = in.size(i);
      p_out_shape[i] = out.size(i);
    }

    for (int i = 0; i < num_inp_dims; i++) {
      p_permute_vec[i] = i;
    }

    p_permute_vec[dim0] = dim1;
    p_permute_vec[dim1] = dim0;

    if (in_type == ScalarType::Float) {
      WORD32* p_inp = (WORD32*)in.const_data_ptr<float>();
      WORD32* p_out = (WORD32*)out.mutable_data_ptr<float>();

      WORD32 ret_val = xa_nn_transpose_32_32(
          p_out,
          p_out_shape,
          p_inp,
          p_inp_shape,
          p_permute_vec,
          num_out_dims,
          num_inp_dims);

      ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

    } else if (in_type == ScalarType::Char) {
      WORD8* p_inp = (WORD8*)in.const_data_ptr<char>();
      WORD8* p_out = (WORD8*)out.mutable_data_ptr<char>();

      WORD32 val = xa_nn_transpose_8_8(
          p_out,
          p_out_shape,
          p_inp,
          p_inp_shape,
          p_permute_vec,
          num_out_dims,
          num_inp_dims);

      ET_KERNEL_CHECK(ctx, val == 0, Internal, out);

    } else if (in_type == ScalarType::Byte) {
      WORD8* p_inp = (WORD8*)in.const_data_ptr<uint8_t>();
      WORD8* p_out = (WORD8*)out.mutable_data_ptr<uint8_t>();

      WORD32 val = xa_nn_transpose_8_8(
          p_out,
          p_out_shape,
          p_inp,
          p_inp_shape,
          p_permute_vec,
          num_out_dims,
          num_inp_dims);

      ET_KERNEL_CHECK(ctx, val == 0, Internal, out);
    }

    return out;
  }

  ET_KERNEL_CHECK(
      ctx,
      check_transpose_copy_args(in, dim0, dim1, out),
      InvalidArgument,
      out);

  ET_SWITCH_ALL_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
    transpose_tensors<CTYPE>(in, dim0, dim1, out);
  });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
