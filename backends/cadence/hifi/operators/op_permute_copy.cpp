/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::runtime::IntArrayRef;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::check_permute_copy_args;
using torch::executor::Error;
using torch::executor::get_permute_copy_out_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

namespace {

void increment_coordinate_permuted(
    const Tensor& tensor,
    size_t* const coordinate,
    IntArrayRef dims) {
  for (int i = dims.size() - 1; i >= 0; i--) {
    size_t d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim();
    coordinate[d]++;
    if (coordinate[d] == tensor.size(d)) {
      coordinate[d] = 0;
    } else {
      return;
    }
  }
}

} // namespace

Tensor& permute_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_permute_copy_args(in, dims, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_permute_copy_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  const auto in_type = out.scalar_type();
  constexpr int kNnlibMaxDim = 16;

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
      p_out_shape[i] = in.size(dims[i]);
      p_permute_vec[i] = dims[i];
    }

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

  size_t in_coord[kTensorDimensionLimit] = {0};
  size_t trailing_dims_memo[kTensorDimensionLimit];
  executorch::runtime::memoizeTrailingDims(in, trailing_dims_memo);

  const char* const in_data = static_cast<const char*>(in.const_data_ptr());
  char* const out_data = static_cast<char*>(out.mutable_data_ptr());
  const size_t element_size = out.element_size();

  for (size_t i = 0; i < out.numel(); ++i) {
    const size_t in_index =
        executorch::runtime::coordinateToIndexWithTrailingDimsMemo(
            in, in_coord, trailing_dims_memo);

    std::memcpy(
        out_data + i * element_size,
        in_data + in_index * element_size,
        element_size);

    increment_coordinate_permuted(in, in_coord, dims);
  }

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
