/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/argmax/arg_reduce_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct ArgReduceParams {
  uint32_t num_rows;
  uint32_t reduce_size;
  uint32_t is_argmin;
  uint32_t _pad;
};

// The cooperative reduction uses var<workgroup> part_val/part_idx arrays of
// width 256; the workgroup size must not exceed that.
static_assert(
    kArgReduceWorkgroupSizeX <= 256,
    "arg_reduce workgroup size exceeds the 256-wide shared partials");

// Last-dim argmax/argmin -> int32 index; mirrors Vulkan arg_reduce_impl.
void arg_reduce_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    uint32_t is_argmin) {
  // aten.argmax/argmin.default args: [in, dim, keepdim, out]
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("arg_reduce: null buffer binding");
  }
  // fp32 input; int index output (the AOT downcasts the int64 index to int32).
  if (in_tensor.is_int || in_tensor.elem_size != 4) {
    throw std::runtime_error("arg_reduce: input must be fp32");
  }
  if (!out_tensor.is_int || out_tensor.elem_size != 4) {
    throw std::runtime_error("arg_reduce: output must be int32 index");
  }

  const int64_t ndim = static_cast<int64_t>(in_tensor.dims.size());
  if (ndim <= 0) {
    throw std::runtime_error("arg_reduce: input must have at least one dim");
  }
  const int64_t dim = graph.get_int(args.at(1));
  if (dim != -1 && dim != ndim - 1) {
    throw std::runtime_error(
        "arg_reduce: only last-dim reduction is supported");
  }

  const uint32_t reduce_size = static_cast<uint32_t>(in_tensor.dims.back());
  const uint32_t num_rows =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(int32_t));
  if (reduce_size == 0u ||
      in_tensor.nbytes / sizeof(float) !=
          static_cast<size_t>(num_rows) * reduce_size) {
    throw std::runtime_error("arg_reduce: shape mismatch (rows * reduce_size)");
  }

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kArgReduceWorkgroupSizeX);
  // One workgroup per row (cooperative reduction); grid = num_rows workgroups.
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, num_rows, 1, "arg_reduce");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ArgReduceParams params = {};
  params.num_rows = num_rows;
  params.reduce_size = reduce_size;
  params.is_argmin = is_argmin;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ArgReduceParams));
  graph.add_uniform_buffer_bytes(sizeof(ArgReduceParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kArgReduceWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ArgReduceParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "arg_reduce",
       workgroup_count.y});

  // Dynamic shapes: recompute reduce_size (last dim) + num_rows + dispatch.
  const bool keepdim = graph.get_bool(args.at(2));
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, keepdim, is_argmin, wg_size, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const uint32_t rsize = static_cast<uint32_t>(d.back());
        if (rsize == 0u) {
          throw std::runtime_error("arg_reduce(resize): zero reduce dim");
        }
        const uint64_t total = utils::numel_of(d);
        const uint32_t rows = static_cast<uint32_t>(total / rsize);
        std::vector<int64_t> od = d;
        if (keepdim) {
          od.back() = 1;
        } else {
          od.pop_back();
        }
        g.set_cur_dims(out_id, od);
        ArgReduceParams p = {};
        p.num_rows = rows;
        p.reduce_size = rsize;
        p.is_argmin = is_argmin;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), rows, 1, "arg_reduce");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

void argmax_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  arg_reduce_impl(graph, args, 0u);
}

void argmin_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  arg_reduce_impl(graph, args, 1u);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.argmax.default, argmax_impl);
  WEBGPU_REGISTER_OP(aten.argmin.default, argmin_impl);
}

} // namespace executorch::backends::webgpu
