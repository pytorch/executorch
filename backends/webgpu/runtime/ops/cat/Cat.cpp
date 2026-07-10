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

    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kCatWGSL,
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
             out_meta_buf,
             sizeof(TensorMeta)},
            {3, WGPUBufferBindingType_Uniform, in_meta_buf, sizeof(TensorMeta)},
            {4, WGPUBufferBindingType_Uniform, params_buf, sizeof(CatParams)},
        },
        &wg_size_constant,
        1);

    graph.add_dispatch({bundle.pipeline, bundle.bind_group, wg_counts[k]});
    // Drop our refs; this input's bind group keeps its uniforms alive.
    wgpuBufferRelease(in_meta_buf);
    wgpuBufferRelease(params_buf);
    off_k += static_cast<uint32_t>(in_tensor.dims[dim]);
  }

  // Drop our ref to the shared out_meta; the bind groups keep it alive.
  wgpuBufferRelease(out_meta_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.cat.default, cat_impl);
}

} // namespace executorch::backends::webgpu
