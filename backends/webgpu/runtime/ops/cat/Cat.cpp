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
#include <executorch/backends/webgpu/runtime/ops/TensorMeta.h>
#include <executorch/backends/webgpu/runtime/ops/cat/cat_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct CatParams {
  uint32_t concat_dim;
  uint32_t off_k;
  uint32_t _pad[2];
};
static_assert(
    sizeof(CatParams) == 16,
    "CatParams must match the WGSL Params uniform (16-byte aligned)");

// cat: 1 dispatch/input -> disjoint out slab at host off_k (Vulkan concat).
void cat_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [tensors (ValueList), dim, out].
  const int list_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(list_id) != WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error("cat: tensors arg is not a ValueList");
  }
  if (graph.get_value_type(args.at(1)) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("cat: dim arg is not a static Int");
  }
  if (graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("cat: out arg is not a tensor");
  }

  WGPUDevice device = graph.device();
  const std::vector<int>& ids = graph.get_value_list(list_id);
  if (ids.empty()) {
    throw std::runtime_error("cat: empty input list");
  }

  const auto& out_tensor = graph.get_tensor(out_id);
  const int ndim = static_cast<int>(out_tensor.dims.size());

  int64_t dim = graph.get_int(args.at(1));
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw std::runtime_error("cat: dim out of range");
  }

  // Workgroup size is invariant across inputs: clamp once, share the constant.
  uint32_t wg_size = utils::clamp_workgroup_size(device, kCatWorkgroupSizeX);

  // Validate + cache input meta/wgc BEFORE any GPU alloc (no leak on throw).
  std::vector<TensorMeta> in_metas(ids.size());
  std::vector<uint32_t> wg_counts(ids.size());
  int64_t concat_sum = 0;
  for (size_t k = 0; k < ids.size(); k++) {
    const int id = ids[k];
    if (graph.get_value_type(id) != WebGPUGraph::ValueType::Tensor) {
      throw std::runtime_error("cat: input list element is not a tensor");
    }
    const auto& in_tensor = graph.get_tensor(id);
    if (static_cast<int>(in_tensor.dims.size()) != ndim) {
      throw std::runtime_error("cat: input rank != output rank");
    }
    for (int d = 0; d < ndim; d++) {
      if (d != dim && in_tensor.dims[d] != out_tensor.dims[d]) {
        throw std::runtime_error("cat: non-concat dim size mismatch");
      }
    }
    fill_tensor_meta(in_tensor, &in_metas[k]);
    if (in_tensor.nbytes !=
        static_cast<size_t>(in_metas[k].numel) * sizeof(float)) {
      throw std::runtime_error("cat: non-fp32 input (nbytes != numel * 4)");
    }
    wg_counts[k] = utils::compute_1d_workgroup_count(
        device, in_metas[k].numel, wg_size, "cat");
    concat_sum += in_tensor.dims[dim];
  }
  if (concat_sum != out_tensor.dims[dim]) {
    throw std::runtime_error("cat: concat dim sizes do not sum to output");
  }

  TensorMeta out_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  if (out_tensor.nbytes !=
      static_cast<size_t>(out_meta.numel) * sizeof(float)) {
    throw std::runtime_error("cat: non-fp32 output (nbytes != numel * 4)");
  }

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  graph.add_uniform_buffer_bytes(sizeof(TensorMeta));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  // Shared shader/layout; fresh pipeline+bind group per input (no double-free).
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kCatWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[5] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Uniform;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  entries[4].binding = 4;
  entries[4].visibility = WGPUShaderStage_Compute;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 5;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  uint32_t off_k = 0;
  for (size_t k = 0; k < ids.size(); k++) {
    const auto& in_tensor = graph.get_tensor(ids[k]);

    CatParams params = {};
    params.concat_dim = static_cast<uint32_t>(dim);
    params.off_k = off_k;

    WGPUBuffer in_meta_buf =
        utils::make_uniform(device, &in_metas[k], sizeof(TensorMeta));
    WGPUBuffer params_buf =
        utils::make_uniform(device, &params, sizeof(CatParams));
    graph.add_uniform_buffer_bytes(sizeof(TensorMeta) + sizeof(CatParams));

    WGPUComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
    pipeline_desc.compute.constantCount = 1;
    pipeline_desc.compute.constants = &wg_size_constant;
    WGPUComputePipeline pipeline =
        wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

    WGPUBindGroupEntry bg_entries[5] = {};
    bg_entries[0].binding = 0;
    bg_entries[0].buffer = in_tensor.buffer;
    bg_entries[0].size = in_tensor.nbytes;
    bg_entries[1].binding = 1;
    bg_entries[1].buffer = out_tensor.buffer;
    bg_entries[1].size = out_tensor.nbytes;
    bg_entries[2].binding = 2;
    bg_entries[2].buffer = out_meta_buf;
    bg_entries[2].size = sizeof(TensorMeta);
    bg_entries[3].binding = 3;
    bg_entries[3].buffer = in_meta_buf;
    bg_entries[3].size = sizeof(TensorMeta);
    bg_entries[4].binding = 4;
    bg_entries[4].buffer = params_buf;
    bg_entries[4].size = sizeof(CatParams);

    WGPUBindGroupDescriptor bg_desc = {};
    bg_desc.layout = bgl;
    bg_desc.entryCount = 5;
    bg_desc.entries = bg_entries;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

    graph.add_dispatch({pipeline, bind_group, wg_counts[k]});
    // Drop our refs; this input's bind group keeps its uniforms alive.
    wgpuBufferRelease(in_meta_buf);
    wgpuBufferRelease(params_buf);
    off_k += static_cast<uint32_t>(in_tensor.dims[dim]);
  }

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Drop our ref to the shared out_meta; the bind groups keep it alive.
  wgpuBufferRelease(out_meta_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.cat.default, cat_impl);
}

} // namespace executorch::backends::webgpu
