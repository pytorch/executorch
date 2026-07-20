/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

// Activation preprocessing operations.
//
// This header collects dispatches that transform activation tensors into
// layouts or dtypes optimized for downstream compute kernels (e.g. quantized
// linear GEMM). Unlike generic view/reshape ops in Transpose.h, these are
// fused transform + cast kernels intended for performance-critical paths.

void add_transpose_cast_contig_to_vectorized_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef output);

} // namespace vkcompute
