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
#include <executorch/backends/webgpu/runtime/ops/linear/linear_tiled_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/linear/linear_vec4_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct LinearParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t has_bias;
};
static_assert(sizeof(LinearParams) == 16, "LinearParams must be 16 bytes");

constexpr uint32_t kTile = 32u;

// aten.linear (+ optional bias); shared-memory tiled GEMM.
void linear_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [input, weight, bias?, out]; out is last. bias (arg 2) is a tensor
  // when present, Null/absent otherwise; add_bias[col] is fused in the shader.
  const int in_id = args.at(0);
  const int w_id = args.at(1);
  const int out_id = args.at(args.size() - 1);
  const bool has_bias = args.size() >= 4 &&
      graph.get_value_type(args.at(2)) == WebGPUGraph::ValueType::Tensor;

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& w = graph.get_tensor(w_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.size() != 2 || w.dims.size() != 2) {
    throw std::runtime_error("WebGPU linear: input/weight must be 2D");
  }
  const uint32_t M = static_cast<uint32_t>(in.dims[0]);
  const uint32_t K = static_cast<uint32_t>(in.dims[1]);
  const uint32_t N = static_cast<uint32_t>(w.dims[0]);
  if (static_cast<uint32_t>(w.dims[1]) != K) {
    throw std::runtime_error("WebGPU linear: weight.dims[1] != input.dims[1]");
  }
  if (M == 0 || N == 0 || K == 0) {
    throw std::runtime_error("WebGPU linear: M, N, or K == 0");
  }

  const uint64_t outputs = static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
  if (in.nbytes != static_cast<uint64_t>(M) * K * sizeof(float) ||
      w.nbytes != static_cast<uint64_t>(N) * K * sizeof(float) ||
      out.nbytes != outputs * sizeof(float)) {
    throw std::runtime_error("WebGPU linear: fp32-only (byte-size mismatch)");
  }

  const uint32_t dispatch_x = (N + kTile - 1u) / kTile;
  const uint32_t dispatch_y = (M + kTile - 1u) / kTile;
  const uint32_t max_wg = utils::queried_max_workgroups(device);
  if (dispatch_x > max_wg || dispatch_y > max_wg) {
    throw std::runtime_error("WebGPU linear: tile grid exceeds dispatch limit");
  }

  if (has_bias) {
    const auto& b = graph.get_tensor(args.at(2));
    if (b.nbytes != static_cast<uint64_t>(N) * sizeof(float)) {
      throw std::runtime_error("WebGPU linear: bias must be [N] fp32");
    }
  }

  LinearParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.has_bias = has_bias ? 1u : 0u;

  // Bias binding (binding 4); a 4-byte dummy satisfies it when None
  // (WGSL-gated).
  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(args.at(2)).buffer : nullptr,
      has_bias ? graph.get_tensor(args.at(2)).nbytes : 0);

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(LinearParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(LinearParams));
  std::memcpy(mapped, &params, sizeof(LinearParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(LinearParams));

  // vec4-over-K path when K%4==0 (scalar out, only K%4, not N%4 like mm).
  const bool use_vec4 = (K % 4u == 0u);
  // Tiled kernel has a fixed @workgroup_size(8, 8, 1) — no override constant.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      use_vec4 ? kLinearVec4WGSL : kLinearTiledWGSL,
      {
          {0, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, w.buffer, w.nbytes},
          {2, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {3,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(LinearParams)},
          {4, WGPUBufferBindingType_ReadOnlyStorage, bias.buffer, bias.nbytes},
      });
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }

  WebGPUDispatch dispatch;
  dispatch.pipeline = bundle.pipeline;
  dispatch.bind_group = bundle.bind_group;
  dispatch.workgroup_count_x = dispatch_x;
  dispatch.workgroup_count_y = dispatch_y;
  dispatch.kernel_name = "linear";
  const size_t dispatch_idx = graph.add_dispatch(dispatch);

  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, M, N, K, dispatch_x, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const uint64_t numel = utils::numel_of(d);
        if (numel % static_cast<uint64_t>(K) != 0u) {
          throw std::runtime_error(
              "WebGPU linear: live input numel not a multiple of K");
        }
        const uint32_t m =
            static_cast<uint32_t>(numel / static_cast<uint64_t>(K));
        if (m == 0u || m > M) {
          throw std::runtime_error(
              "WebGPU linear: live M is 0 or exceeds the build-time max");
        }
        LinearParams p = {};
        p.M = m;
        p.N = N;
        p.K = K;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x = dispatch_x;
        g.dispatch_at(dispatch_idx).workgroup_count_y =
            (m + kTile - 1u) / kTile;
        g.set_cur_dims(
            out_id, {static_cast<int64_t>(m), static_cast<int64_t>(N)});
      });

  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.linear.default, linear_impl);
}

} // namespace executorch::backends::webgpu
