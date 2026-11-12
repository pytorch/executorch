/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <optional>

namespace vkcompute {

/*
 * A Winograd F(m x m, r x r) configuration: it computes m x m output tiles of a
 * convolution with an odd r x r kernel, stride 1, dilation 1 and padding
 * (r - 1) / 2 from (m + r - 1) x (m + r - 1) input tiles. Its transform and
 * weight packing shaders are the variants with suffix f<m>x<m>_<r>x<r>; every
 * configuration shares the GEMM shaders.
 */
struct Conv2dWinogradConfig {
  int64_t out_tile;
  int64_t kernel_size;

  int64_t in_tile() const {
    return out_tile + kernel_size - 1;
  }

  // Elements of a transformed tile
  int64_t tile_elements() const {
    return in_tile() * in_tile();
  }

  // Output tiles along a dimension of `size` pixels
  int64_t tiles_of(const int64_t size) const {
    return utils::div_up(size, out_tile);
  }
};

constexpr Conv2dWinogradConfig kConv2dWinogradF4x4_3x3{4, 3};

// Whether conv2d_winograd_impl() supports a convolution with `config`: an
// ungrouped, batch-1 Half convolution with the kernel size, stride, dilation
// and padding of `config` on channels-packed Texture3D tensors, whose scratch
// textures fit the device limits.
bool can_use_conv2d_winograd(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups);

// The first Winograd configuration that supports a convolution, whether or not
// it is faster than the other conv2d routes.
std::optional<Conv2dWinogradConfig> find_conv2d_winograd_config(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups);

// The Winograd configuration that computes a convolution faster than the other
// conv2d routes, if any.
std::optional<Conv2dWinogradConfig> pick_conv2d_winograd_config(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups);

/*
 * Computes a convolution that can_use_conv2d_winograd() accepts for `config`,
 * in three dispatches per chunk of output tile rows:
 *
 *   1. conv2d_winograd_input transforms each input tile d into V = B^T d B.
 *   2. conv2d_winograd_gemm multiplies each element of V, over the input
 *      channels, with the matching element of the transformed weight
 *      U = G g G^T, which a prepack node computes on the GPU.
 *   3. conv2d_winograd_output transforms each product M into the output tile
 *      A^T M A, adds the bias and applies the epilogue.
 *
 * V and M are scratch tensors that hold one chunk; chunks bound their size.
 *
 * `chunk_rows_override` and `weight_storage_override` force the tile rows per
 * chunk and the storage of the transformed weight instead of choosing them;
 * tests use them to cover every chunk and GEMM variant on one device.
 */
void conv2d_winograd_impl(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef out,
    const Conv2dEpilogue& epilogue,
    const std::optional<int64_t> chunk_rows_override = std::nullopt,
    const std::optional<utils::StorageType> weight_storage_override =
        std::nullopt);

} // namespace vkcompute
