// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include "value_spec.h"

#include <cstdint>

namespace executorch {
namespace vulkan {
namespace prototyping {

ValueRef quantized_weights_canvas(
    ComputeGraph& graph,
    const ValueRef weight_ref);

ValueRef float_tensor_canvas(ComputeGraph& graph, const ValueRef weight_ref);

// Compute weight sums for quantized operations (linear and convolution)
void compute_weight_sums(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_features,
    int64_t elements_per_output_feature);

// Compute weight sums for 4D quantized conv2d operations
// Weight layout: [C_out, K_h, K_w, align_up_4(C_in_per_group)]
void compute_weight_sums_4d(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t aligned_in_channels);

// Compute weight sums for 4-bit group symmetric quantized weights
void compute_weight_sums_4bit_grouped(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t num_groups,
    int64_t out_features,
    int64_t group_size);

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
