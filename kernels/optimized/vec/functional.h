/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cpu/vec/vec.h>

namespace executorch::vec {
// This function implements broadcasting binary operation on two tensors
// where lhs tensor is treated to be of shape [outer_size, broadcast_size, inner_size]
// and rhs tensor is treated to be of shape [outer_size, 1, inner_size]
// And this 1st dimension is considered broadcasting dimension
// This formula can map broadcasting on any dim=broadcast_dim
// for any two N dimensional tensors, where 0 < braodcast_dim < N-1
template <typename scalar_t, typename Op>
inline void broadcasting_map_3d_and_unsqueezed_3d(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* lhs,
    const scalar_t* rhs,
    int64_t outer_size,
    int64_t broadcast_size,
    int64_t inner_size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t outer_stride_lhs = inner_size * broadcast_size;
  int64_t outer_stride_rhs = inner_size;
  int64_t broadcast_stride_lhs = inner_size;
  for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    const scalar_t* lhs_outer = lhs + outer_idx * outer_stride_lhs;
    scalar_t* output_data_row = output_data + outer_idx * outer_stride_lhs;
    const scalar_t* rhs_outer = rhs + outer_idx * outer_stride_rhs;
    for (int64_t broadcast_idx = 0; broadcast_idx < broadcast_size; ++broadcast_idx) {
      const scalar_t* lhs_outer_2 = lhs_outer + broadcast_idx * broadcast_stride_lhs;
      scalar_t* output_data_row_2 = output_data_row + broadcast_idx * broadcast_stride_lhs;
      int64_t inner_idx = 0;
      for (; inner_idx < inner_size - (inner_size % Vec::size()); inner_idx += Vec::size()) {
        Vec data_vec = Vec::loadu(lhs_outer_2 + inner_idx);
        Vec data_vec2 = Vec::loadu(rhs_outer + inner_idx);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data_row_2 + inner_idx);
      }
      if (inner_size - inner_idx > 0) {
        Vec data_vec = Vec::loadu(lhs_outer_2 + inner_idx, inner_size - inner_idx);
        Vec data_vec2 = Vec::loadu(rhs_outer + inner_idx, inner_size - inner_idx);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data_row_2 + inner_idx, inner_size - inner_idx);
      }
    }
  }
}

template <typename scalar_t, typename Op>
inline void broadcasting_map_2d_by_1d(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size,
    int64_t size2) {
  broadcasting_map_3d_and_unsqueezed_3d(vec_fun, output_data, input_data, input_data2, 1, size, size2);
}

/*
Following function is used to implement broadcasting binary operation on two tensors
where lhs tensor is treated to be of shape [outer_size, broadcast_size] and
rhs tensor is treated to be of shape [outer_size, 1]
Any two N dimensional tensors can be mapped to this formula
when lhs size = [lhs0, lhs1, ..., lhsN-1] and rhs size = [rhs0, rhs1, ..., 1]
by viewing the two tensors as
lhs size = [lsh0 * lsh1 * ... * lshN-2, lhsN-1]
rhs size = [rsh0 * rsh1 * ... * rshN-2, 1]
*/
template <typename scalar_t, typename Op>
inline void broadcasting_map_broadcast_last_dim(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* lhs,
    const scalar_t* rhs,
    int64_t outer_size,
    int64_t broadcast_size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t outer_stride_lhs = broadcast_size;
  for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    const scalar_t* lhs_outer = lhs + outer_idx * outer_stride_lhs;
    scalar_t* output_data_row = output_data + outer_idx * outer_stride_lhs;
    int64_t inner_idx = 0;
    Vec data_vec2 = Vec(rhs[outer_idx]);
    for (; inner_idx < broadcast_size - (broadcast_size % Vec::size()); inner_idx += Vec::size()) {
      Vec data_vec = Vec::loadu(lhs_outer + inner_idx);
      Vec output_vec = vec_fun(data_vec, data_vec2);
      output_vec.store(output_data_row + inner_idx);
    }
    if (broadcast_size - inner_idx > 0) {
      Vec data_vec = Vec::loadu(lhs_outer + inner_idx, broadcast_size - inner_idx);
      Vec output_vec = vec_fun(data_vec, data_vec2);
      output_vec.store(output_data_row + inner_idx, broadcast_size - inner_idx);
    }
  }
}
} // namespace executorch::vec
