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

/**
 * Extracts a value at index i from an int array. If the array length is 1, then
 * the first element will be returned regardless of what i is requested to
 * simulate broadcasting.
 */
int64_t val_at(IntArrayRef array, size_t i, int64_t default_value = 1);

/**
 * Checks that all elements of an IntArray are greater than or equal to `val`.
 */
bool int_array_all_ge(IntArrayRef array, int64_t val);

bool stride_is_valid(IntArrayRef stride, size_t kernel_ndim);

bool padding_is_valid(
    IntArrayRef padding,
    IntArrayRef kernel_size,
    size_t kernel_ndim,
    bool enforce_half_kernel = false);

bool dilation_is_valid(IntArrayRef dilation, size_t kernel_ndim);

bool output_size_is_valid(
    exec_aten::ArrayRef<exec_aten::SizesType> output_size);

void get_unsqueezed_sizes(
    const Tensor& t,
    int64_t unsqueeze_dim,
    exec_aten::SizesType* sizes_arr,
    size_t& ndim);

void get_unsqueezed_dim_order(
    const Tensor& t,
    exec_aten::DimOrderType unsqueeze_dim,
    exec_aten::DimOrderType* dim_order_arr);

/**
 * Given an input tensor and N-dim kernel parameters, calculates the output size
 * of the N-dim kernel region.
 */
void calculate_kernel_output_sizes(
    const Tensor& in,
    IntArrayRef kernel_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    bool ceil_mode = false);

//
// Operator specific utility functions
//

bool check_convolution_args(
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    Tensor& out);

void get_convolution_out_target_size(
    const Tensor& in,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim);

} // namespace executor
} // namespace torch
