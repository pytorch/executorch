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
#include <executorch/backends/webgpu/runtime/ops/native_group_norm/group_norm_reduce_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/native_group_norm/group_norm_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct GroupNormParams {
  uint32_t n_channels;
  uint32_t hxw;
  uint32_t num_groups;
  uint32_t chans_per_group;
  uint32_t numel;
  uint32_t mean_numel;
  uint32_t group_size;
  float eps;
};
static_assert(
    sizeof(GroupNormParams) == 32,
    "GroupNormParams must match the WGSL Params struct (32 bytes)");

// The reduce shader's cooperative reduction uses var<workgroup> f32[256]
// arrays; the workgroup size (used for both passes) must not exceed that width.
static_assert(
    kGroupNormWorkgroupSizeX <= 256,
    "group_norm workgroup size exceeds the 256-wide shared reduction arrays");

struct GnBinding {
  WGPUBuffer buffer;
  uint64_t size;
  WGPUBufferBindingType type;
};

// Build one compute dispatch from a binding list (last = uniform).
size_t add_gn_dispatch(
    WebGPUGraph& graph,
    WGPUDevice device,
    const char* wgsl_code,
    uint32_t wg_size,
    const std::vector<GnBinding>& binds,
    utils::WgCount wgc,
    const char* label) {
  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  std::vector<utils::BindingSpec> specs(binds.size());
  for (size_t i = 0; i < binds.size(); i++) {
    specs[i] = {
        static_cast<uint32_t>(i),
        binds[i].type,
        binds[i].buffer,
        binds[i].size};
  }

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device, wgsl_code, specs, &wg_size_constant, 1);

  const size_t idx = graph.add_dispatch(
      {bundle.pipeline, bundle.bind_group, wgc.x, label, wgc.y});
  return idx;
}

// native_group_norm: per-group reduce + per-channel norm (Vulkan GroupNorm).
void native_group_norm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [in, weight, bias, N, C, HxW, group, eps, out_tuple].
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int group_id = args.at(6);
  const int eps_id = args.at(7);
  const int out_list_id = args.at(8);

  if (graph.get_value_type(out_list_id) != WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error("group_norm: out arg is not a value list");
  }
  const std::vector<int>& out_ids = graph.get_value_list(out_list_id);
  if (out_ids.size() != 3) {
    throw std::runtime_error("group_norm: expected 3 outputs (out/mean/rstd)");
  }
  const int out_id = out_ids.at(0);
  const int mean_id = out_ids.at(1);
  const int rstd_id = out_ids.at(2);

  const int tensor_ids[6] = {
      in_id, weight_id, bias_id, out_id, mean_id, rstd_id};
  for (int id : tensor_ids) {
    if (graph.get_value_type(id) != WebGPUGraph::ValueType::Tensor) {
      throw std::runtime_error("group_norm: in/weight/bias/out not a tensor");
    }
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& bias_tensor = graph.get_tensor(bias_id);
  const auto& mean_tensor = graph.get_tensor(mean_id);
  const auto& rstd_tensor = graph.get_tensor(rstd_id);

  if (in_tensor.dims.size() != 4) {
    throw std::runtime_error("group_norm: only 4D (NCHW) input is supported");
  }
  const uint32_t n_batch = static_cast<uint32_t>(in_tensor.dims.at(0));
  const uint32_t n_channels = static_cast<uint32_t>(in_tensor.dims.at(1));
  const uint32_t hxw =
      static_cast<uint32_t>(in_tensor.dims.at(2) * in_tensor.dims.at(3));
  const int64_t group = graph.get_int(group_id);
  if (group <= 0 || n_channels % static_cast<uint32_t>(group) != 0) {
    throw std::runtime_error("group_norm: C not divisible by group");
  }
  const uint32_t num_groups = static_cast<uint32_t>(group);
  const uint32_t chans_per_group = n_channels / num_groups;

  const uint64_t numel = static_cast<uint64_t>(n_batch) * n_channels * hxw;
  const uint32_t mean_numel = n_batch * num_groups;
  if (in_tensor.nbytes != numel * sizeof(float) ||
      out_tensor.nbytes != numel * sizeof(float)) {
    throw std::runtime_error("group_norm: fp32-only (byte-size mismatch)");
  }
  const size_t chan_bytes = static_cast<size_t>(n_channels) * sizeof(float);
  if (weight_tensor.nbytes != chan_bytes || bias_tensor.nbytes != chan_bytes) {
    throw std::runtime_error("group_norm: weight/bias length != num_channels");
  }
  const size_t mean_bytes = static_cast<size_t>(mean_numel) * sizeof(float);
  if (mean_tensor.nbytes != mean_bytes || rstd_tensor.nbytes != mean_bytes) {
    throw std::runtime_error("group_norm: mean/rstd size != N * group");
  }

  float eps;
  if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Double) {
    eps = static_cast<float>(graph.get_double(eps_id));
  } else if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Int) {
    eps = static_cast<float>(graph.get_int(eps_id));
  } else {
    throw std::runtime_error("group_norm: unexpected eps value type");
  }

  GroupNormParams params = {};
  params.n_channels = n_channels;
  params.hxw = hxw;
  params.num_groups = num_groups;
  params.chans_per_group = chans_per_group;
  params.numel = static_cast<uint32_t>(numel);
  params.mean_numel = mean_numel;
  params.group_size = chans_per_group * hxw;
  params.eps = eps;

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kGroupNormWorkgroupSizeX);
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(GroupNormParams));
  graph.add_uniform_buffer_bytes(sizeof(GroupNormParams));

  const WGPUBufferBindingType ro = WGPUBufferBindingType_ReadOnlyStorage;
  const WGPUBufferBindingType rw = WGPUBufferBindingType_Storage;
  const WGPUBufferBindingType uni = WGPUBufferBindingType_Uniform;

  // Pass 1: reduce -> mean/rstd (one workgroup per (n, group), cooperative).
  utils::WgCount reduce_wgc =
      utils::compute_2d_workgroup_count(device, mean_numel, 1, "gn_reduce");
  const size_t reduce_idx = add_gn_dispatch(
      graph,
      device,
      kGroupNormReduceWGSL,
      wg_size,
      {{in_tensor.buffer, in_tensor.nbytes, ro},
       {mean_tensor.buffer, mean_tensor.nbytes, rw},
       {rstd_tensor.buffer, rstd_tensor.nbytes, rw},
       {params_buf, sizeof(GroupNormParams), uni}},
      reduce_wgc,
      "group_norm_reduce");

  // Pass 2: normalize (execute() runs one pass/dispatch, so reduce precedes).
  utils::WgCount norm_wgc = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(numel), wg_size, "gn_norm");
  const size_t norm_idx = add_gn_dispatch(
      graph,
      device,
      kGroupNormWGSL,
      wg_size,
      {{in_tensor.buffer, in_tensor.nbytes, ro},
       {out_tensor.buffer, out_tensor.nbytes, rw},
       {weight_tensor.buffer, weight_tensor.nbytes, ro},
       {bias_tensor.buffer, bias_tensor.nbytes, ro},
       {mean_tensor.buffer, mean_tensor.nbytes, ro},
       {rstd_tensor.buffer, rstd_tensor.nbytes, ro},
       {params_buf, sizeof(GroupNormParams), uni}},
      norm_wgc,
      "group_norm");

  // Dynamic shapes: recompute params + both dispatch counts from the live dims.
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       mean_id,
       rstd_id,
       num_groups,
       eps,
       wg_size,
       reduce_idx,
       norm_idx,
       p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() != 4) {
          throw std::runtime_error("group_norm(resize): input is not 4D");
        }
        const uint32_t nb = static_cast<uint32_t>(d[0]);
        const uint32_t c = static_cast<uint32_t>(d[1]);
        const uint32_t hw = static_cast<uint32_t>(d[2] * d[3]);
        if (c % num_groups != 0) {
          throw std::runtime_error(
              "group_norm(resize): C not divisible by group");
        }
        const uint32_t dpg = c / num_groups;
        const uint64_t numel64 = static_cast<uint64_t>(nb) * c * hw;
        const uint64_t group_size64 = static_cast<uint64_t>(dpg) * hw;
        if (numel64 > std::numeric_limits<uint32_t>::max() ||
            group_size64 > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("group_norm(resize): numel exceeds uint32");
        }
        GroupNormParams p = {};
        p.n_channels = c;
        p.hxw = hw;
        p.num_groups = num_groups;
        p.chans_per_group = dpg;
        p.numel = static_cast<uint32_t>(numel64);
        p.mean_numel = nb * num_groups;
        p.group_size = static_cast<uint32_t>(group_size64);
        p.eps = eps;
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount rwgc = utils::compute_2d_workgroup_count(
            g.device(), p.mean_numel, 1, "gn_reduce(resize)");
        g.dispatch_at(reduce_idx).workgroup_count_x = rwgc.x;
        g.dispatch_at(reduce_idx).workgroup_count_y = rwgc.y;
        const utils::WgCount nwgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "gn_norm(resize)");
        g.dispatch_at(norm_idx).workgroup_count_x = nwgc.x;
        g.dispatch_at(norm_idx).workgroup_count_y = nwgc.y;
        g.set_cur_dims(out_id, d);
        const std::vector<int64_t> mr = {
            d[0], static_cast<int64_t>(num_groups)};
        g.set_cur_dims(mean_id, mr);
        g.set_cur_dims(rstd_id, mr);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.native_group_norm.default, native_group_norm_impl);
}

} // namespace executorch::backends::webgpu
