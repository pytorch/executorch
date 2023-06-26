// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/util/tensor_util.h>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>

namespace torch {
namespace executor {

//
// Reduction
//

/**
 * For `size` elements of `data_in`, accumulates the modified values into a
 * value using `reduce_fun`, and returns the accumulated value. The `stride` can
 * also be defined; by default it is set to 1.
 */
template <typename CTYPE, typename ReduceOp>
inline CTYPE apply_unary_reduce_fn(
    const ReduceOp& reduce_fun,
    const CTYPE* const data_in,
    const int64_t size,
    const int64_t stride = 1) {
  CTYPE acc_val = data_in[0];
  for (size_t i = 1; i < size; i++) {
    acc_val = reduce_fun(data_in[i * stride], acc_val);
  }
  return acc_val;
}

//
// Mapping
//

/**
 * Applies `map_fun` to `size` elements of `data_in`, writing results to
 * `data_out`. The `stride` can also be defined; by default it is set to 1.
 */
template <typename CTYPE_IN, typename CTYPE_OUT, typename MapOp>
inline void apply_unary_map_fn(
    const MapOp& map_fun,
    const CTYPE_IN* const data_in,
    CTYPE_OUT* const data_out,
    const int64_t size,
    const int64_t stride = 1) {
  for (size_t i = 0; i < size; i++) {
    data_out[i * stride] = map_fun(data_in[i * stride]);
  }
}

//
// Mapping + Reduction
//

/**
 * Applies `map_fun` to `size` elements of `data_in`, accumulates the modified
 * values into a value using `reduce_fun`, and returns the accumulated value.
 * The `stride` can also be defined; by default it is set to 1.
 */
template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename MapOp,
    typename ReduceOp>
inline CTYPE_OUT apply_unary_map_reduce_fn(
    const MapOp& map_fun,
    const ReduceOp& reduce_fun,
    const CTYPE_IN* const data_in,
    const int64_t size,
    const int64_t stride = 1) {
  CTYPE_OUT acc_val = map_fun(data_in[0]);
  for (size_t i = 1; i < size; ++i) {
    acc_val = reduce_fun(map_fun(data_in[i * stride]), acc_val);
  }
  return acc_val;
}

//
// Mapping with broadcasting
//

/**
 * Useful for binary elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 */
template <typename CTYPE_A, typename CTYPE_B, typename CTYPE_OUT, typename Op>
inline void apply_binary_elementwise_fn(
    const Op& compute_fun,
    const Tensor& a,
    const CTYPE_A* const data_a,
    const Tensor& b,
    const CTYPE_B* const data_b,
    const Tensor& out,
    CTYPE_OUT* const data_out) {
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool any_is_broadcasted = (a_is_broadcasted || b_is_broadcasted);

  for (size_t i = 0; i < out.numel(); ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out, a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out, b);
      }
    }

    data_out[i] = compute_fun(data_a[a_linear_index], data_b[b_linear_index]);
  }
}

/**
 * Useful for ternary elementwise operators. For each element of the inputs,
 * perform a computation and write to the corresponding element of the output.
 * Tensor broadcasting is applied wherever it is required.
 */
template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_C,
    typename CTYPE_OUT,
    typename Op>
inline void apply_ternary_elementwise_fn(
    const Op& compute_fun,
    const Tensor& a,
    const CTYPE_A* const data_a,
    const Tensor& b,
    const CTYPE_B* const data_b,
    const Tensor& c,
    const CTYPE_C* const data_c,
    const Tensor& out,
    CTYPE_OUT* const data_out) {
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool c_is_broadcasted = !out.sizes().equals(c.sizes());
  const bool any_is_broadcasted =
      (a_is_broadcasted || b_is_broadcasted || c_is_broadcasted);

  for (size_t i = 0; i < out.numel(); ++i) {
    size_t a_linear_index = i;
    size_t b_linear_index = i;
    size_t c_linear_index = i;

    if (any_is_broadcasted) {
      size_t out_indexes[kTensorDimensionLimit];
      delinearize_index(i, out, out_indexes, kTensorDimensionLimit);

      if (a_is_broadcasted) {
        a_linear_index = linearize_access_indexes(out_indexes, out, a);
      }
      if (b_is_broadcasted) {
        b_linear_index = linearize_access_indexes(out_indexes, out, b);
      }
      if (c_is_broadcasted) {
        c_linear_index = linearize_access_indexes(out_indexes, out, c);
      }
    }

    data_out[i] = compute_fun(
        data_a[a_linear_index], data_b[b_linear_index], data_c[c_linear_index]);
  }
}

} // namespace executor
} // namespace torch
