/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>
#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensor_is_default_dim_order;
using executorch::runtime::tensors_have_same_dim_order;
using torch::executor::check_mm_args;
using torch::executor::Error;
using torch::executor::get_mm_out_target_size;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& mm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mat2,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, check_mm_args(in, mat2, out), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_mm_out_target_size(in, mat2, output_sizes, &output_ndim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, mat2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ScalarType out_type = out.scalar_type();

  constexpr auto name = "mm.out";

  bool optimized = true;

  if (out_type != ScalarType::Float)
    optimized = false;

  if (optimized) {
    const float* in_data = in.const_data_ptr<float>();
    const float* mat2_data = mat2.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    int64_t m = in.size(0);
    int64_t n = in.size(1);

    int64_t p = mat2.size(1);

    WORD32 rows = m;
    WORD32 cols1 = n;
    WORD32 row_stride1 = n;
    WORD32 vec_count = p;
    WORD32 vec_offset = n;
    WORD32 out_offset = 1;
    WORD32 out_stride = p;

    WORD32* __restrict__ p_o =
        (WORD32* __restrict__)kernels::allocate_temp_memory(
            ctx, (n * p) * sizeof(WORD32));

    // Allocate zero-initialized bias for matmul function (it doesn't accept
    // NULL)
    FLOAT32* __restrict__ p_bias_zero =
        (FLOAT32* __restrict__)kernels::allocate_temp_memory(
            ctx, m * sizeof(FLOAT32));

    // Initialize bias to zero since mm operation has no bias
    memset(p_bias_zero, 0, m * sizeof(FLOAT32));

    WORD32 p_inp_shape[2];
    p_inp_shape[0] = n;
    p_inp_shape[1] = p;

    WORD32 p_out_shape[2];
    p_out_shape[0] = p;
    p_out_shape[1] = n;

    WORD32 p_permute_vec[2] = {1, 0};

    WORD32 num_out_dims = 2;
    WORD32 num_inp_dims = 2;

    const FLOAT32* __restrict__ p_mat1 = in_data;
    const FLOAT32* __restrict__ p_vec1 = mat2_data;
    FLOAT32* __restrict__ p_out = out_data;

    WORD32* p_inp = (WORD32*)p_vec1;

    WORD32 t = xa_nn_transpose_32_32(
        p_o,
        p_out_shape,
        p_inp,
        p_inp_shape,
        p_permute_vec,
        num_out_dims,
        num_inp_dims);

    const FLOAT32* __restrict__ p_vec = (const FLOAT32* __restrict__)p_o;

    // mm will always be converted to addmm and to linear, and move transpose to
    // graph
    WORD32 val = xa_nn_matmul_f32xf32_f32(
        p_out,
        p_mat1,
        p_vec,
        p_bias_zero,
        rows,
        cols1,
        row_stride1,
        vec_count,
        vec_offset,
        out_offset,
        out_stride);
    return out;
  }

  ET_SWITCH_REAL_TYPES_AND2(
      Half, BFloat16, in.scalar_type(), ctx, name, CTYPE, [&]() {
        size_t m = in.size(0);
        size_t n = in.size(1);
        size_t p = mat2.size(1);

        torch::executor::vec_matmul<CTYPE>(
            out.mutable_data_ptr<CTYPE>(),
            in.const_data_ptr<CTYPE>(),
            mat2.const_data_ptr<CTYPE>(),
            m,
            n,
            p);
      });

  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
