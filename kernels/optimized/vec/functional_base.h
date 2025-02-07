/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>

namespace executorch {
namespace vec {

// slow path
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    at::vec::Vectorized<scalar_t> acc_vec,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  scalar_t acc_arr[Vec::size()];
  acc_vec.store(acc_arr);
  for (int64_t i = 1; i < size; ++i) {
    std::array<scalar_t, Vec::size()> acc_arr_next = {0};
    acc_arr_next[0] = acc_arr[i];
    Vec acc_vec_next = Vec::loadu(acc_arr_next.data());
    acc_vec = vec_fun(acc_vec, acc_vec_next);
  }
  acc_vec.store(acc_arr);
  return acc_arr[0];
}

template <typename scalar_t, typename Op>
struct VecReduceAllSIMD {
  static inline scalar_t apply(const Op& vec_fun, const at::vec::Vectorized<scalar_t>& acc_vec) {
    return vec_reduce_all(vec_fun, acc_vec, at::vec::Vectorized<scalar_t>::size());
  }
};

#if defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) && !defined(C10_MOBILE)
#if defined(CPU_CAPABILITY_AVX2)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(const Op& vec_fun, const at::vec::Vectorized<float>& acc_vec) {
    using Vec = at::vec::Vectorized<float>;
    Vec v = acc_vec;
    // 128-bit shuffle
    Vec v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm256_cvtss_f32(v);
  }
};
#endif // defined(CPU_CAPABILITY_AVX2)
#if defined(CPU_CAPABILITY_AVX512)
template <typename Op>
struct VecReduceAllSIMD<float, Op> {
  static inline float apply(const Op& vec_fun, const at::vec::Vectorized<float>& acc_vec) {
    using Vec = at::vec::Vectorized<float>;
    Vec v = acc_vec;
    // 256-bit shuffle
    Vec v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = vec_fun(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = vec_fun(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = vec_fun(v, v1);
    return _mm512_cvtss_f32(v);
  }
};
#endif // defined(CPU_CAPABILITY_AVX512)
#endif // defined(__GNUC__) && (__GNUC__ > 5) && !defined(_MSC_VER) && !defined(C10_MOBILE)

template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(const Op& vec_fun, const at::vec::Vectorized<scalar_t>& acc_vec) {
  return VecReduceAllSIMD<scalar_t, Op>::apply(vec_fun, acc_vec);
}

template <typename scalar_t, typename Op>
inline scalar_t reduce_all(const Op& vec_fun, const scalar_t* data, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
  int64_t d = Vec::size();
  Vec acc_vec = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec = vec_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(vec_fun, acc_vec);
}

// similar to reduce_all, but reduces into two outputs
template <typename scalar_t, typename Op1, typename Op2>
inline std::pair<scalar_t, scalar_t> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const scalar_t* data, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    auto loaded_data = Vec::loadu(data, size);
    return std::pair<scalar_t, scalar_t>(
      vec_reduce_all(vec_fun1, loaded_data, size),
      vec_reduce_all(vec_fun2, loaded_data, size));
  }
  int64_t d = Vec::size();
  Vec acc_vec1 = Vec::loadu(data);
  Vec acc_vec2 = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec1 = vec_fun1(acc_vec1, data_vec);
    acc_vec2 = vec_fun2(acc_vec2, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec1 = Vec::set(acc_vec1, vec_fun1(acc_vec1, data_vec), size - d);
    acc_vec2 = Vec::set(acc_vec2, vec_fun2(acc_vec2, data_vec), size - d);
  }
  return std::pair<scalar_t, scalar_t>(
    vec_reduce_all(vec_fun1, acc_vec1),
    vec_reduce_all(vec_fun2, acc_vec2));
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(red_fun, map_fun(Vec::loadu(data, size)), size);
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    data_vec = map_fun(data_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    data_vec = map_fun(data_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    data_vec = map_fun(data_vec, data2_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    const scalar_t* data3,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    Vec data3_vec = Vec::loadu(data3, size);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }

  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    Vec data3_vec = Vec::loadu(data3 + d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    Vec data3_vec = Vec::loadu(data3 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec);
}

template <typename scalar_t, typename Op>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d));
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(input_data + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(input_data + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map3(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map4(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    const scalar_t* input_data4,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec data_vec4 = Vec::loadu(input_data4 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec data_vec4 = Vec::loadu(input_data4 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d, size - d);
  }
}


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

} // namespace vec
} // namespace executorch
