/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_grid_sampler_2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef grid = args.at(1).refs.at(1);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const std::vector<int64_t> grid_sizes = graph->sizes_of(grid);

  // input  : [N, C, Hin, Win]
  // grid   : [N, Hout, Wout, 2]
  // output : [N, C, Hout, Wout]
  std::vector<int64_t> out_sizes = {
      in_sizes.at(0), in_sizes.at(1), grid_sizes.at(1), grid_sizes.at(2)};

  graph->virtual_resize(out, out_sizes);
}

void add_grid_sampler_2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef grid,
    const ValueRef interpolation_mode,
    const ValueRef padding_mode,
    const ValueRef align_corners,
    const ValueRef out) {
  // Runtime sanity checks. The Python partitioner is supposed to filter out
  // unsupported configurations, but guard against bypass paths here too.
  // mode: 0 = bilinear, 1 = nearest, 2 = bicubic
  VK_CHECK_COND(
      graph.extract_scalar<int64_t>(interpolation_mode) == 0,
      "Vulkan grid_sampler_2d only supports bilinear interpolation");
  // padding_mode: 0 = zeros, 1 = border, 2 = reflection
  VK_CHECK_COND(
      graph.extract_scalar<int64_t>(padding_mode) == 1,
      "Vulkan grid_sampler_2d only supports border padding");
  VK_CHECK_COND(
      graph.get_bool(align_corners),
      "Vulkan grid_sampler_2d requires align_corners=true");

  // Defense-in-depth layout validation. The partitioner enforces these
  // layouts via `inputs_storage` in op_registry.py::register_grid_sampler_2d,
  // but the shader hard-codes channels-packed texture indexing for in/out and
  // contiguous buffer indexing for grid, so a layout mismatch here would be a
  // silent miscompute. Per the etvk-implement-operator skill ("Validate
  // tensor layout assumptions"), assert these explicitly.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(in),
      "Vulkan grid_sampler_2d requires input to be a channels-packed texture");
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan grid_sampler_2d requires output to be a channels-packed texture");
  VK_CHECK_COND(
      graph.is_contiguous_buffer_tensor(grid),
      "Vulkan grid_sampler_2d requires grid to be a contiguous buffer");

  // The shader binds t_in, t_out, and t_grid with a single DTYPE selected via
  // `dtype_of(out)` below. The op registry allows `grid` to be fp16 or fp32
  // independently of the input dtype, so without this guard a mixed-precision
  // model (e.g., fp32 flow grid + fp16 activations) would bind the fp32 grid
  // buffer as half and silently miscompute. Op tests use matching dtypes for
  // all args, so they would not catch this.
  VK_CHECK_COND(
      graph.dtype_of(grid) == graph.dtype_of(out),
      "Vulkan grid_sampler_2d requires grid and input to share dtype");

  std::string kernel_name("grid_sampler_2d");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, grid}, vkapi::kRead}},
      // Shader params buffers. `meta_ubo` packs sizes, limits, axis_map, and
      // packed_dim into the canonical TextureMetadata struct (see vtensor.md);
      // the shader derives Wout/Hout/N/num_z_per_n from `outp.sizes` and
      // `outp.limits`, so no extra params buffer is needed.
      {graph.meta_ubo(out), graph.meta_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants — pass the output tensor's hashed layout so
      // the shader can specialize on packed_dim at pipeline creation time.
      {graph.hashed_layout_of(out)},
      // Resize Args
      {},
      // Resizing Logic
      resize_grid_sampler_2d_node));
}

void grid_sampler_2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Argument order matches kernels/portable/cpu/op_grid_sampler_2d.cpp:
  //   (input, grid, interpolation_mode, padding_mode, align_corners, out)
  return add_grid_sampler_2d_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.grid_sampler_2d.default, grid_sampler_2d);
}

} // namespace vkcompute
