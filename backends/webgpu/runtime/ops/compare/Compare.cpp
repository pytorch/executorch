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
#include <executorch/backends/webgpu/runtime/ops/compare/compare_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct CompareParams {
  uint32_t num_elements;
  uint32_t op;
  uint32_t _pad[2];
};
static_assert(sizeof(CompareParams) == 16, "CompareParams must be 16 bytes");

// Elementwise fp32 compare -> bool (op 0=eq..4=ge); eq is torch-exact a==b.
void compare_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    uint32_t op) {
  // args: [in1, in2, out]; fp32 in, bool out; out=args.back().
  const int in1_id = args.at(0);
  const int in2_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in1_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(in2_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("compare: in1/in2/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in1_tensor = graph.get_tensor(in1_id);
  const auto& in2_tensor = graph.get_tensor(in2_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in1_tensor.buffer == nullptr || in2_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("compare: null buffer binding");
  }
  // fp32 inputs; bool output (1-byte int-typed, NOT int8-quantized).
  if (in1_tensor.is_int || in2_tensor.is_int || in1_tensor.elem_size != 4 ||
      in2_tensor.elem_size != 4) {
    throw std::runtime_error("compare: fp32 inputs only");
  }
  if (!out_tensor.is_int || out_tensor.elem_size != 1) {
    throw std::runtime_error("compare: output must be a 1-byte bool tensor");
  }
  const uint64_t numel = out_tensor.nbytes;
  // out bool packed 4/word (array<u32>); numel%4==0 gates the readback map.
  if (numel == 0u || numel % 4u != 0u || numel > UINT32_MAX) {
    throw std::runtime_error("compare: numel must be a nonzero mult of 4");
  }
  const uint64_t in_numel = in1_tensor.nbytes / sizeof(float);
  if (in1_tensor.nbytes != in2_tensor.nbytes || in_numel != numel) {
    throw std::runtime_error("compare: in/out numel mismatch (same-shape)");
  }

  CompareParams params = {};
  params.num_elements = static_cast<uint32_t>(numel);
  params.op = op;

  const uint32_t words = static_cast<uint32_t>(numel / 4u);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kCompareWorkgroupSizeX);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, words, wg_size, "compare");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(CompareParams));
  graph.add_uniform_buffer_bytes(sizeof(CompareParams));

  // out (rw storage) + in1/in2 (ro storage) + params (uniform).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kCompareWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           in1_tensor.buffer,
           in1_tensor.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           in2_tensor.buffer,
           in2_tensor.nbytes},
          {3,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(CompareParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "compare",
       workgroup_count.y});

  // Dynamic shapes: recompute numel/dispatch; out follows in1 (same-shape).
  WGPUBuffer params_buf = uniform_buffer;
  auto resize = [in1_id, in2_id, out_id, op, wg_size, dispatch_idx, params_buf](
                    WebGPUGraph& g) {
    const auto& d = g.cur_dims(in1_id);
    const uint64_t n = utils::numel_of(d);
    if (n == 0u || n % 4u != 0u || n > UINT32_MAX ||
        utils::numel_of(g.cur_dims(in2_id)) != n) {
      throw std::runtime_error("compare(resize): numel must be a mult of 4");
    }
    g.set_cur_dims(out_id, d);
    CompareParams p = {};
    p.num_elements = static_cast<uint32_t>(n);
    p.op = op;
    wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
    const utils::WgCount wgc = utils::compute_2d_workgroup_count(
        g.device(), static_cast<uint32_t>(n / 4u), wg_size, "compare");
    g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
    g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
  };
  graph.add_tensor_resize_hook(in1_id, resize);
  graph.add_tensor_resize_hook(in2_id, resize);

  graph.own_uniform_buffer(uniform_buffer);
}

void eq_op(WebGPUGraph& graph, const std::vector<int>& args) {
  compare_impl(graph, args, 0u);
}
void lt_op(WebGPUGraph& graph, const std::vector<int>& args) {
  compare_impl(graph, args, 1u);
}
void le_op(WebGPUGraph& graph, const std::vector<int>& args) {
  compare_impl(graph, args, 2u);
}
void gt_op(WebGPUGraph& graph, const std::vector<int>& args) {
  compare_impl(graph, args, 3u);
}
void ge_op(WebGPUGraph& graph, const std::vector<int>& args) {
  compare_impl(graph, args, 4u);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.eq.Tensor, eq_op);
  WEBGPU_REGISTER_OP(aten.lt.Tensor, lt_op);
  WEBGPU_REGISTER_OP(aten.le.Tensor, le_op);
  WEBGPU_REGISTER_OP(aten.gt.Tensor, gt_op);
  WEBGPU_REGISTER_OP(aten.ge.Tensor, ge_op);
}

} // namespace executorch::backends::webgpu
