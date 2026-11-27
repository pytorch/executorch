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
#include <executorch/backends/webgpu/runtime/ops/avg_pool2d/avg_pool2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct AvgPoolParams {
  uint32_t kh;
  uint32_t kw;
  uint32_t sh;
  uint32_t sw;
  uint32_t ph;
  uint32_t pw;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t channels;
  uint32_t numel;
  int32_t divisor_override;
  uint32_t count_include_pad;
  uint32_t has_divisor_override;
  uint32_t pad1;
};
static_assert(sizeof(AvgPoolParams) == 64, "AvgPoolParams must be 64 bytes");

// Pooled output extent (in+2p-k)/s+1; ceil_mode: ceil + drop pad-only window.
uint32_t
pool_out_dim(int64_t in, int64_t k, int64_t s, int64_t p, bool ceil_mode) {
  const int64_t num = in + 2 * p - k;
  int64_t o = ceil_mode ? (num + s - 1) / s + 1 : num / s + 1;
  if (ceil_mode && (o - 1) * s >= in + p) {
    o -= 1;
  }
  if (o < 0) {
    o = 0;
  }
  return static_cast<uint32_t>(o);
}

// Read an IntList[2]: empty->fallback (stride->kernel), len1->broadcast, len2.
void read_pair(
    const std::vector<int64_t>& v,
    uint32_t fallback_h,
    uint32_t fallback_w,
    uint32_t* h,
    uint32_t* w) {
  if (v.empty()) {
    *h = fallback_h;
    *w = fallback_w;
  } else if (v.size() == 1) {
    *h = static_cast<uint32_t>(v[0]);
    *w = static_cast<uint32_t>(v[0]);
  } else {
    *h = static_cast<uint32_t>(v[0]);
    *w = static_cast<uint32_t>(v[1]);
  }
}

// avg_pool2d: average over a KxK window per output cell (Vulkan Pool.cpp).
void avg_pool2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [in, kernel, stride, padding, ceil_mode, cip, divisor, out].
  const int in_id = args.at(0);
  const int kernel_id = args.at(1);
  const int stride_id = args.at(2);
  const int padding_id = args.at(3);
  const int ceil_id = args.at(4);
  const int cip_id = args.at(5);
  const int divisor_id = args.at(6);
  const int out_id = args.at(7);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("avg_pool2d: in/out arg is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.dims.size() != 4 || out_tensor.dims.size() != 4) {
    throw std::runtime_error("avg_pool2d: only 4D (NCHW) tensors supported");
  }

  uint32_t kh, kw, sh, sw, ph, pw;
  read_pair(graph.get_int_list(kernel_id), 1, 1, &kh, &kw);
  read_pair(graph.get_int_list(stride_id), kh, kw, &sh, &sw);
  read_pair(graph.get_int_list(padding_id), 0, 0, &ph, &pw);

  int32_t divisor_override = 0;
  uint32_t has_divisor_override = 0u;
  if (graph.get_value_type(divisor_id) == WebGPUGraph::ValueType::Int) {
    divisor_override = static_cast<int32_t>(graph.get_int(divisor_id));
    has_divisor_override = 1u;
  }
  const uint32_t count_include_pad = graph.get_bool(cip_id) ? 1u : 0u;

  const uint32_t channels = static_cast<uint32_t>(in_tensor.dims.at(1));
  const uint32_t in_h = static_cast<uint32_t>(in_tensor.dims.at(2));
  const uint32_t in_w = static_cast<uint32_t>(in_tensor.dims.at(3));
  const uint32_t out_h = static_cast<uint32_t>(out_tensor.dims.at(2));
  const uint32_t out_w = static_cast<uint32_t>(out_tensor.dims.at(3));

  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes != out_numel * sizeof(float)) {
    throw std::runtime_error("avg_pool2d: fp32-only (byte-size mismatch)");
  }
  if (out_numel > UINT32_MAX) {
    throw std::runtime_error("avg_pool2d: output numel exceeds u32");
  }

  AvgPoolParams params = {};
  params.kh = kh;
  params.kw = kw;
  params.sh = sh;
  params.sw = sw;
  params.ph = ph;
  params.pw = pw;
  params.in_h = in_h;
  params.in_w = in_w;
  params.out_h = out_h;
  params.out_w = out_w;
  params.channels = channels;
  params.numel = static_cast<uint32_t>(out_numel);
  params.divisor_override = divisor_override;
  params.count_include_pad = count_include_pad;
  params.has_divisor_override = has_divisor_override;

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kAvgPool2dWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "avg_pool2d");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(AvgPoolParams));
  graph.add_uniform_buffer_bytes(sizeof(AvgPoolParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kAvgPool2dWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2, WGPUBufferBindingType_Uniform, params_buf, sizeof(AvgPoolParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "avg_pool2d",
       workgroup_count.y});

  // Dynamic shapes: recompute the pooled output extent + params + dispatch.
  const bool ceil_mode = graph.get_bool(ceil_id);
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       kh,
       kw,
       sh,
       sw,
       ph,
       pw,
       divisor_override,
       has_divisor_override,
       count_include_pad,
       ceil_mode,
       wg_size,
       dispatch_idx,
       p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() != 4) {
          throw std::runtime_error("avg_pool2d(resize): input is not 4D");
        }
        AvgPoolParams p = {};
        p.kh = kh;
        p.kw = kw;
        p.sh = sh;
        p.sw = sw;
        p.ph = ph;
        p.pw = pw;
        p.in_h = static_cast<uint32_t>(d[2]);
        p.in_w = static_cast<uint32_t>(d[3]);
        p.out_h = pool_out_dim(d[2], kh, sh, ph, ceil_mode);
        p.out_w = pool_out_dim(d[3], kw, sw, pw, ceil_mode);
        p.channels = static_cast<uint32_t>(d[1]);
        const uint64_t out_numel =
            static_cast<uint64_t>(d[0]) * d[1] * p.out_h * p.out_w;
        if (out_numel > UINT32_MAX) {
          throw std::runtime_error(
              "avg_pool2d(resize): output numel exceeds u32");
        }
        p.numel = static_cast<uint32_t>(out_numel);
        p.divisor_override = divisor_override;
        p.has_divisor_override = has_divisor_override;
        p.count_include_pad = count_include_pad;
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "avg_pool2d(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        const std::vector<int64_t> out_d = {
            d[0],
            d[1],
            static_cast<int64_t>(p.out_h),
            static_cast<int64_t>(p.out_w)};
        g.set_cur_dims(out_id, out_d);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.avg_pool2d.default, avg_pool2d_impl);
}

} // namespace executorch::backends::webgpu
