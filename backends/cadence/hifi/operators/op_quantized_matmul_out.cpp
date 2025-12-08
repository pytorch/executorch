/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <on_device_ai/Assistant/Jarvis/min_runtime/operators/generic/op_quantized_matmul.h>
#include <stdlib.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using torch::executor::RuntimeContext;

namespace impl {
namespace HiFi {
namespace native {

// The quantized matmul. The quantized matmul accumulates in a wider register,
// whose type is TA.

template <typename T>
void inline _typed_quantized_matmul(
    RuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const exec_aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  T* __restrict__ out_data = out.mutable_data_ptr<T>();
  const T* __restrict__ X_data = X.const_data_ptr<T>();
  const T* __restrict__ Y_data = Y.const_data_ptr<T>();
  size_t batch_size = getLeadingDims(X, X.dim() - 2);
  size_t leading_dim = X.size(X.dim() - 2);
  size_t out_dim = Y.size(Y.dim() - 1 - transposed);
  size_t in_dim = X.size(X.dim() - 1);

  const int32_t* __restrict__ bias_data =
      (WORD32* __restrict__)kernels::allocate_temp_memory(
          ctx, (leading_dim * in_dim) * sizeof(int32_t));

  ET_CHECK_MSG(bias_data != nullptr, "MemoryAllocationFailed");

  std::memset((void*)bias_data, 0, (leading_dim * in_dim) * sizeof(int32_t));

  uint8_t* y_data_temp = NULL;

  if (!transposed) {
    y_data_temp =
        (uint8_t*)kernels::allocate_temp_memory(ctx, (leading_dim * in_dim));

    ET_CHECK_MSG(y_data_temp != nullptr, "MemoryAllocationFailed");
  }

  for (size_t i = 0; i < batch_size; ++i) {
    const T* x = X_data + i * leading_dim * in_dim;
    const T* y = Y_data + i * in_dim * out_dim;
    T* z = out_data + i * leading_dim * out_dim;
    if (transposed) {
      if (out.scalar_type() == exec_aten::ScalarType::Byte) {
        WORD32 ret_val = xa_nn_matmul_asym8uxasym8u_asym8u(
            (uint8_t*)z, // p_out
            (uint8_t*)y, // p_mat1,
            (uint8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias

        ET_CHECK_MSG(ret_val == 0, "An internal error occured");
      } else {
        WORD32 ret_val = xa_nn_matmul_asym8sxasym8s_asym8s(
            (int8_t*)z, // p_out
            (int8_t*)y, // p_mat1,
            (int8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias

        ET_CHECK_MSG(ret_val == 0, "An internal error occured");
      }
    } else {
      /* Assuming matmul is 2D always */
      WORD32 num_inp_dims = 2;
      WORD32 num_out_dims = 2;

      WORD32 p_inp_shape[2];
      WORD32 p_out_shape[2];
      WORD32 p_permute_vec[2] = {1, 0};

      p_inp_shape[0] = leading_dim;
      p_inp_shape[1] = in_dim;
      p_out_shape[0] = in_dim;
      p_out_shape[1] = leading_dim;

      WORD32 ret_val = xa_nn_transpose_8_8(
          (int8_t*)y_data_temp,
          p_out_shape,
          (int8_t*)y,
          p_inp_shape,
          p_permute_vec,
          num_out_dims,
          num_inp_dims);

      ET_CHECK_MSG(ret_val == 0, "An internal error occured");

      if (out.scalar_type() == exec_aten::ScalarType::Byte) {
        WORD32 ret_val = xa_nn_matmul_asym8uxasym8u_asym8u(
            (uint8_t*)z, // p_out
            (uint8_t*)y_data_temp, // p_mat1,
            (uint8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias

        ET_CHECK_MSG(ret_val == 0, "An internal error occured");
      } else {
        WORD32 ret_val = xa_nn_matmul_asym8sxasym8s_asym8s(
            (int8_t*)z, // p_out
            (int8_t*)y_data_temp, // p_mat1,
            (int8_t*)x, // p_mat2,
            bias_data, // p_bias
            out_dim, // rows of p_mat1
            in_dim, // cols of p_mat1
            in_dim, // row_stride of p_mat1
            leading_dim, // vec_count, i.e., rows of p_mat2
            in_dim, // vec_offset of p_mat2.
            out_dim, // out_offset, i.e., offset of next output element written
            1, // out_stride, i.e., stride to go to next output row
            -(static_cast<int32_t>(Y_zero_point)), // mat1_zero_bias
            -(static_cast<int32_t>(X_zero_point)), // mat2_zero_bias
            static_cast<int32_t>(out_multiplier), // out_multiplier
            static_cast<int32_t>(out_shift), // out_shift
            static_cast<int32_t>(out_zero_point)); // out_zero_bias

        ET_CHECK_MSG(ret_val == 0, "An internal error occured");
      }
    }
  }
}

void quantized_matmul_out(
    RuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const exec_aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  size_t batch_size = getLeadingDims(X, X.dim() - 2);
  size_t leading_dim = X.size(X.dim() - 2);
  size_t out_dim = Y.size(Y.dim() - 1 - transposed);
  size_t in_dim = X.size(X.dim() - 1);
  if (out.scalar_type() == exec_aten::ScalarType::Short) {
    ::impl::generic::native::quantized_matmul_out(
        ctx,
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  } else if (out.scalar_type() == exec_aten::ScalarType::Byte) {
    _typed_quantized_matmul<uint8_t>(
        ctx,
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);

  } else if (out.scalar_type() == exec_aten::ScalarType::Char) {
    _typed_quantized_matmul<int8_t>(
        ctx,
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(out.scalar_type()));
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
