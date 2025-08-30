/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_addmm_args(
    const Tensor& in,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

bool check_bmm_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_bmm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_mm_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_mm_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

bool check_linear_args(const Tensor& in, const Tensor& mat2, Tensor& out);

void get_linear_out_target_size(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

namespace internal {

template <typename CTYPE>
void bmm_out_impl(const Tensor& in, const Tensor& mat2, Tensor& out) {
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  const CTYPE* mat2_data = mat2.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  int64_t batch_size = in.size(0);
  int64_t m = in.size(1);
  int64_t n = in.size(2);
  int64_t p = mat2.size(2);

  for (int b = 0; b < batch_size; ++b) {
    const CTYPE* in_data_offset = in_data + b * m * n;
    const CTYPE* mat2_data_offset = mat2_data + b * n * p;
    CTYPE* out_data_offset = out_data + b * m * p;

    for (const auto i : c10::irange(m)) {
      for (const auto j : c10::irange(p)) {
        CTYPE sum = static_cast<CTYPE>(0.0);
        for (const auto k : c10::irange(n)) {
          sum += in_data_offset[i * n + k] * mat2_data_offset[k * p + j];
        }
        out_data_offset[i * p + j] = sum;
      }
    }
  }
}

} // namespace internal
} // namespace executor
} // namespace torch
