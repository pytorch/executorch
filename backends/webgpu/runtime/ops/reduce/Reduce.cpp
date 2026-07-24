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
#include <executorch/backends/webgpu/runtime/ops/reduce/reduce_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned).
struct ReduceParams {
  uint32_t outer;
  uint32_t r;
  uint32_t inner;
  uint32_t is_mean;
};
static_assert(sizeof(ReduceParams) == 16, "ReduceParams must be 16 bytes");

// Normalize + validate CONTIGUOUS reduced dims, then fold the tensor into
// [outer, r, inner] where r spans the reduced range. The kernel reduces r
// elements per (outer, inner) output — identical for 1 or N contiguous dims
// (e.g. global avg pool over [H,W] -> r = H*W). Non-contiguous multi-dim reduce
// is rejected (rare; would need a separate gather).
void decompose_dims(
    const std::vector<int64_t>& dims,
    std::vector<int64_t> reduce_dims,
    uint32_t& outer,
    uint32_t& r,
    uint32_t& inner) {
  const int64_t ndim = static_cast<int64_t>(dims.size());
  if (ndim == 0 || reduce_dims.empty()) {
    throw std::runtime_error("WebGPU reduce: dim out of range");
  }
  for (int64_t& d : reduce_dims) {
    if (d < 0) {
      d += ndim;
    }
    if (d < 0 || d >= ndim) {
      throw std::runtime_error("WebGPU reduce: dim out of range");
    }
  }
  std::sort(reduce_dims.begin(), reduce_dims.end());
  for (size_t i = 1; i < reduce_dims.size(); ++i) {
    if (reduce_dims[i] == reduce_dims[i - 1]) {
      throw std::runtime_error("WebGPU reduce: duplicate reduced dim");
    }
    if (reduce_dims[i] != reduce_dims[i - 1] + 1) {
      throw std::runtime_error(
          "WebGPU reduce: only contiguous reduced dims supported");
    }
  }
  const int64_t first_rd = reduce_dims.front();
  const int64_t last_rd = reduce_dims.back();
  uint64_t o = 1, rr = 1, in = 1;
  for (int64_t d = 0; d < first_rd; ++d) {
    o *= static_cast<uint64_t>(dims[d]);
  }
  for (int64_t d = first_rd; d <= last_rd; ++d) {
    rr *= static_cast<uint64_t>(dims[d]);
  }
  for (int64_t d = last_rd + 1; d < ndim; ++d) {
    in *= static_cast<uint64_t>(dims[d]);
  }
  outer = static_cast<uint32_t>(o);
  r = static_cast<uint32_t>(rr);
  inner = static_cast<uint32_t>(in);
}

void reduce_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    bool is_mean,
    const char* op_name) {
  const int in_id = args.at(0);
  const int dim_id = args.at(1);
  const int keepdim_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  bool keepdim = false;
  if (graph.get_value_type(keepdim_id) == WebGPUGraph::ValueType::Int) {
    keepdim = graph.get_int(keepdim_id) != 0;
  }

  if (in.dims.empty()) {
    throw std::runtime_error("WebGPU reduce: scalar input unsupported");
  }
  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error("WebGPU reduce: dim arg is not an IntList");
  }
  // Contiguous multi-dim reduction (e.g. global avg pool over [H,W]) folds into
  // one [outer, r, inner]; single-dim is the r = one-dim case.
  const std::vector<int64_t> reduce_dims = graph.get_int_list(dim_id);
  if (reduce_dims.empty()) {
    throw std::runtime_error("WebGPU reduce: empty dim list");
  }

  uint32_t outer = 0, r = 0, inner = 0;
  decompose_dims(in.dims, reduce_dims, outer, r, inner);
  if (outer == 0 || r == 0 || inner == 0) {
    throw std::runtime_error("WebGPU reduce: zero-sized reduction");
  }

  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint64_t outputs = static_cast<uint64_t>(outer) * inner;
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != outputs * sizeof(float)) {
    throw std::runtime_error("WebGPU reduce: fp32-only (byte-size mismatch)");
  }
  if (outputs > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU reduce: output count exceeds dispatch limit");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kReduceWorkgroupSizeX);
  // Cooperative reduction: one workgroup per output element (2D-folded grid).
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(outputs), 1u, op_name);

  ReduceParams params = {};
  params.outer = outer;
  params.r = r;
  params.inner = inner;
  params.is_mean = is_mean ? 1u : 0u;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(ReduceParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(ReduceParams));
  std::memcpy(mapped, &params, sizeof(ReduceParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(ReduceParams));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kReduceWGSL,
      {
          {0, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {1, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ReduceParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       op_name,
       workgroup_count.y});

  // Dynamic shapes: recompute the decomposition for the reduced dim + dispatch.
  WGPUBuffer params_buf = uniform_buffer;
  const uint32_t is_mean_u = is_mean ? 1u : 0u;
  const uint64_t build_outputs = outputs;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       reduce_dims,
       keepdim,
       is_mean_u,
       build_outputs,
       dispatch_idx,
       params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        uint32_t o = 0, rr = 0, n = 0;
        decompose_dims(
            std::vector<int64_t>(d.begin(), d.end()), reduce_dims, o, rr, n);
        if (o == 0u || rr == 0u || n == 0u) {
          throw std::runtime_error("WebGPU reduce: live zero-sized reduction");
        }
        const uint64_t live_outputs = static_cast<uint64_t>(o) * n;
        if (live_outputs > build_outputs) {
          throw std::runtime_error(
              "WebGPU reduce: live output count exceeds build max");
        }
        ReduceParams p = {};
        p.outer = o;
        p.r = rr;
        p.inner = n;
        p.is_mean = is_mean_u;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(),
            static_cast<uint32_t>(live_outputs),
            1u,
            "reduce(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        // Propagate reduced output dims for downstream resize hooks.
        int64_t nd = static_cast<int64_t>(d.size());
        std::vector<bool> is_rd(static_cast<size_t>(nd), false);
        for (int64_t rd : reduce_dims) {
          const int64_t n2 = rd < 0 ? rd + nd : rd;
          if (n2 >= 0 && n2 < nd) {
            is_rd[static_cast<size_t>(n2)] = true;
          }
        }
        std::vector<int64_t> od;
        for (int64_t i = 0; i < nd; ++i) {
          if (is_rd[static_cast<size_t>(i)]) {
            if (keepdim) {
              od.push_back(1);
            }
          } else {
            od.push_back(d[i]);
          }
        }
        g.set_cur_dims(out_id, od);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

void sum_dim_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  reduce_impl(graph, args, /*is_mean=*/false, "sum.dim_IntList");
}

void mean_dim_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  reduce_impl(graph, args, /*is_mean=*/true, "mean.dim");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sum.dim_IntList, sum_dim_impl);
  WEBGPU_REGISTER_OP(aten.mean.dim, mean_dim_impl);
}

} // namespace executorch::backends::webgpu
