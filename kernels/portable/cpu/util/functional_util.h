/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

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

} // namespace executor
} // namespace torch
