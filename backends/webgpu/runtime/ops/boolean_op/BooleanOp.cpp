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
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_eq_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_ge_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_gt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_le_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_lt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/compare_ne_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/boolean_op/logical_not_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Shared uniform for the byte-packed-bool op family. `scalar` is unused (0) for
// the scalar-less (unary) logical_not variant.
struct BoolOpParams {
  uint32_t num_elements;
  float scalar;
  uint32_t _pad1;
  uint32_t _pad2;
};

float read_scalar(WebGPUGraph& graph, int id, const char* op_name) {
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::Double) {
    return static_cast<float>(graph.get_double(id));
  }
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::Int) {
    return static_cast<float>(graph.get_int(id));
  }
  throw std::runtime_error(std::string(op_name) + ": scalar is not int/double");
}

// Dispatch a byte-packed-bool op (scalar compares + logical_not): read an input
// buffer, write a u32 output packing 4 bool bytes per word, one thread per word
// (no inter-thread write race). The caller validates dtypes/shapes and supplies
// `numel` (logical element count = prod(dims)), the scalar (0 when unused), and
// the per-variant shader from boolean_op.yaml.
void dispatch_bool_op(
    WebGPUGraph& graph,
    int self_id,
    int out_id,
    uint32_t numel,
    float scalar,
    const char* wgsl,
    uint32_t wg_size_x,
    const char* op_name) {
  WGPUDevice device = graph.device();
  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  // The output (and logical_not's packed-bool input) is a u32 array; round the
  // binding up to whole words even when the byte count isn't a multiple of 4.
  const size_t self_bind_size = (self_tensor.nbytes + 3) & ~size_t(3);
  const size_t out_bind_size = (out_tensor.nbytes + 3) & ~size_t(3);
  const uint32_t n_words = (numel + 3u) / 4u;

  uint32_t wg_size = utils::clamp_workgroup_size(device, wg_size_x);
  uint32_t workgroup_count =
      utils::compute_1d_workgroup_count(device, n_words, wg_size, op_name);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  BoolOpParams params = {numel, scalar, 0u, 0u};
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(BoolOpParams));
  graph.add_uniform_buffer_bytes(sizeof(BoolOpParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[3] = {};
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].buffer.type = WGPUBufferBindingType_Uniform;
  for (uint32_t i = 0; i < 3; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
  }

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 3;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[3] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = self_tensor.buffer;
  bg_entries[0].size = self_bind_size;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_bind_size;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = params_buf;
  bg_entries[2].size = sizeof(BoolOpParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx =
      graph.add_dispatch({pipeline, bind_group, workgroup_count});

  WGPUBuffer p_buf = params_buf;
  auto resize =
      [self_id, out_id, scalar, wg_size, dispatch_idx, p_buf, op_name](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(self_id);
        uint32_t n = 1u;
        for (auto x : d) {
          n *= static_cast<uint32_t>(x);
        }
        g.set_cur_dims(out_id, d);
        BoolOpParams p = {n, scalar, 0u, 0u};
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const uint32_t nw = (n + 3u) / 4u;
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(g.device(), nw, wg_size, op_name);
      };
  graph.add_tensor_resize_hook(self_id, resize);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

// cmp(self[i], scalar) -> byte-packed bool. args: [self, scalar, out].
void compare_dispatch(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    const char* wgsl,
    uint32_t wg_size_x,
    const char* op_name) {
  const int self_id = args.at(0);
  const int out_id = args.at(args.size() - 1);
  const float scalar = read_scalar(graph, args.at(1), op_name);

  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (self_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error(std::string(op_name) + ": null buffer binding");
  }
  if (self_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error(std::string(op_name) + ": self is not fp32");
  }
  const uint32_t numel =
      static_cast<uint32_t>(self_tensor.nbytes / sizeof(float));
  if (out_tensor.nbytes != static_cast<size_t>(numel)) {
    throw std::runtime_error(
        std::string(op_name) + ": out is not a 1-byte (bool) tensor");
  }
  dispatch_bool_op(
      graph, self_id, out_id, numel, scalar, wgsl, wg_size_x, op_name);
}

void eq_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareEqWGSL, kCompareEqWorkgroupSizeX, "eq.Scalar");
}
void ne_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareNeWGSL, kCompareNeWorkgroupSizeX, "ne.Scalar");
}
void le_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareLeWGSL, kCompareLeWorkgroupSizeX, "le.Scalar");
}
void ge_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareGeWGSL, kCompareGeWorkgroupSizeX, "ge.Scalar");
}
void lt_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareLtWGSL, kCompareLtWorkgroupSizeX, "lt.Scalar");
}
void gt_scalar_impl(WebGPUGraph& g, const std::vector<int>& a) {
  compare_dispatch(g, a, kCompareGtWGSL, kCompareGtWorkgroupSizeX, "gt.Scalar");
}

// logical_not: byte-packed bool -> byte-packed bool. args: [self, out].
void logical_not_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int self_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (self_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("logical_not: null buffer binding");
  }
  if (out_tensor.nbytes != self_tensor.nbytes) {
    throw std::runtime_error("logical_not: self/out byte-size mismatch");
  }
  const uint32_t numel = static_cast<uint32_t>(self_tensor.nbytes);
  dispatch_bool_op(
      graph,
      self_id,
      out_id,
      numel,
      0.0f,
      kLogicalNotWGSL,
      kLogicalNotWorkgroupSizeX,
      "logical_not");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.eq.Scalar, eq_scalar_impl);
  WEBGPU_REGISTER_OP(aten.ne.Scalar, ne_scalar_impl);
  WEBGPU_REGISTER_OP(aten.le.Scalar, le_scalar_impl);
  WEBGPU_REGISTER_OP(aten.ge.Scalar, ge_scalar_impl);
  WEBGPU_REGISTER_OP(aten.lt.Scalar, lt_scalar_impl);
  WEBGPU_REGISTER_OP(aten.gt.Scalar, gt_scalar_impl);
  WEBGPU_REGISTER_OP(aten.logical_not.default, logical_not_impl);
}

} // namespace executorch::backends::webgpu
