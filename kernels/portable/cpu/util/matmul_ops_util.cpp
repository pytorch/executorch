/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/matmul_ops_util.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

bool check_addmm_args(
    const Tensor& in,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(mat1, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(mat2, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, 2));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mat1, mat2));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_size_at_dims(mat1, 1, mat2, 0));

  return true;
}

bool check_bmm_args(const Tensor& in, const Tensor& mat2, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, 3));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(mat2, 3));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, 3));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mat2, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_size_at_dims(in, 0, mat2, 0));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_size_at_dims(in, 2, mat2, 1));

  return true;
}

void get_bmm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = 3;
  out_sizes[0] = mat1.size(0);
  out_sizes[1] = mat1.size(1);
  out_sizes[2] = mat2.size(2);
}

bool check_mm_args(const Tensor& in, const Tensor& mat2, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(mat2, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, 2));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mat2, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_size_at_dims(in, 1, mat2, 0));

  return true;
}

bool check_linear_args(const Tensor& in, const Tensor& mat2, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() == out.dim());
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() >= 2);
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(mat2, 2));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, mat2, out));

  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_size_at_dims(in, in.dim() - 1, mat2, 1));

  return true;
}

void get_mm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = 2;
  out_sizes[0] = mat1.size(0);
  out_sizes[1] = mat2.size(1);
}

void get_linear_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = mat1.dim();
  for (int ii = 0; ii < mat1.dim() - 1; ++ii) {
    out_sizes[ii] = mat1.sizes()[ii];
  }
  out_sizes[mat1.dim() - 1] = mat2.size(0);
}

} // namespace executor
} // namespace torch
