/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ComputeGraph side of the cooperative-matrix family kernels
// (CoopmatFamily.h), shared by the q4gsw and dq8ca_q4gsw node builders.

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/CoopmatFamily.h>

namespace vkcompute {

// The family key inputs of the graph's device.
CoopmatDeviceKey coopmat_device_key(ComputeGraph* graph);

WmmaStorage to_wmma_storage(utils::StorageType storage);

// The storage prepack_quantized_linear_weight() picks for a 4-bit weight:
// texture2d unless the packed extents exceed the device limit.
utils::StorageType predicted_q4_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data);

// One workgroup per WG_TILE_M x WG_TILE_N output tile for a family kernel;
// quantized_linear_gwg() for any other kernel name. args.at(0).refs.at(0) is
// the output.
GlobalWorkGrid wmma_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

} // namespace vkcompute
