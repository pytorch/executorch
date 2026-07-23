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
#include <executorch/backends/webgpu/runtime/ops/bmm/bmm_tiled_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/bmm/bmm_vec4_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct BmmParams {
  uint32_t B;
  uint32_t M;
  uint32_t N;
  uint32_t K;
};
static_assert(sizeof(BmmParams) == 16, "BmmParams must be 16 bytes");

constexpr uint32_t kTile = 32u;

// Batched shared-memory tiled GEMM.
void bmm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int a_id = args.at(0);
  const int b_id = args.at(1);
  const int out_id = args.at(2);

  WGPUDevice device = graph.device();

  const auto& a = graph.get_tensor(a_id);
  const auto& b = graph.get_tensor(b_id);
  const auto& out = graph.get_tensor(out_id);

  if (a.dims.size() != 3 || b.dims.size() != 3) {
    throw std::runtime_error("WebGPU bmm: inputs must be 3D");
  }
  const uint32_t B = static_cast<uint32_t>(a.dims[0]);
  const uint32_t M = static_cast<uint32_t>(a.dims[1]);
  const uint32_t K = static_cast<uint32_t>(a.dims[2]);
  const uint32_t N = static_cast<uint32_t>(b.dims[2]);
  if (static_cast<uint32_t>(b.dims[0]) != B ||
      static_cast<uint32_t>(b.dims[1]) != K) {
    throw std::runtime_error("WebGPU bmm: batch/K mismatch between a and b");
  }
  if (B == 0 || M == 0 || N == 0 || K == 0) {
    throw std::runtime_error("WebGPU bmm: B, M, N, or K == 0");
  }

  const uint64_t outputs =
      static_cast<uint64_t>(B) * static_cast<uint64_t>(M) * N;
  if (a.nbytes != static_cast<uint64_t>(B) * M * K * sizeof(float) ||
      b.nbytes != static_cast<uint64_t>(B) * K * N * sizeof(float) ||
      out.nbytes != outputs * sizeof(float)) {
    throw std::runtime_error("WebGPU bmm: fp32-only (byte-size mismatch)");
  }

  const uint32_t max_wg = utils::queried_max_workgroups(device);
  const uint32_t dispatch_x = (N + kTile - 1u) / kTile;
  // uint64 so B * ceil(M/32) can't wrap before the limit check.
  const uint64_t dispatch_y64 =
      static_cast<uint64_t>(B) * ((M + kTile - 1u) / kTile);
  if (dispatch_x > max_wg || dispatch_y64 > max_wg) {
    throw std::runtime_error("WebGPU bmm: tile grid exceeds dispatch limit");
  }
  const uint32_t dispatch_y = static_cast<uint32_t>(dispatch_y64);

  BmmParams params = {};
  params.B = B;
  params.M = M;
  params.N = N;
  params.K = K;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(BmmParams));
  graph.add_uniform_buffer_bytes(sizeof(BmmParams));

  // vec4 path when K and N are multiples of 4 (wider 16B loads); else scalar.
  const bool use_vec4 = (K % 4u == 0u) && (N % 4u == 0u);
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {use_vec4 ? kBmmVec4WGSL : kBmmTiledWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  // Tiled kernel has a fixed @workgroup_size(8, 8, 1) — no override constant.
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[4] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = a.buffer;
  bg_entries[0].size = a.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = b.buffer;
  bg_entries[1].size = b.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out.buffer;
  bg_entries[2].size = out.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(BmmParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WebGPUDispatch dispatch;
  dispatch.pipeline = pipeline;
  dispatch.bind_group = bind_group;
  dispatch.workgroup_count_x = dispatch_x;
  dispatch.workgroup_count_y = dispatch_y;
  dispatch.kernel_name = "bmm";
  const size_t dispatch_idx = graph.add_dispatch(dispatch);

  // Dynamic shapes: re-derive B, M (from a) and N (from b) + the tile grid.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      a_id,
      [a_id, b_id, out_id, B, M, N, K, use_vec4, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& da = g.cur_dims(a_id);
        const auto& db = g.cur_dims(b_id);
        if (da.size() != 3 || db.size() != 3 ||
            static_cast<uint32_t>(da[2]) != K ||
            static_cast<uint32_t>(db[1]) != K ||
            static_cast<uint32_t>(db[0]) != static_cast<uint32_t>(da[0])) {
          throw std::runtime_error("WebGPU bmm: live input dims invalid");
        }
        const uint32_t live_b = static_cast<uint32_t>(da[0]);
        const uint32_t live_m = static_cast<uint32_t>(da[1]);
        const uint32_t live_n = static_cast<uint32_t>(db[2]);
        if (live_b == 0u || live_m == 0u || live_n == 0u) {
          throw std::runtime_error("WebGPU bmm: live B, M, or N == 0");
        }
        if (live_b > B || live_m > M || live_n > N) {
          throw std::runtime_error(
              "WebGPU bmm: live dims exceed build-time max");
        }
        // vec4 baked for N%4==0; a live N breaking it reads past the row.
        if (use_vec4 && live_n % 4u != 0u) {
          throw std::runtime_error("WebGPU bmm: live N breaks vec4 alignment");
        }
        BmmParams p = {};
        p.B = live_b;
        p.M = live_m;
        p.N = live_n;
        p.K = K;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const uint32_t max_wg = utils::queried_max_workgroups(g.device());
        const uint32_t dx = (live_n + kTile - 1u) / kTile;
        const uint64_t dy64 =
            static_cast<uint64_t>(live_b) * ((live_m + kTile - 1u) / kTile);
        if (dx > max_wg || dy64 > max_wg) {
          throw std::runtime_error(
              "WebGPU bmm(resize): tile grid exceeds dispatch limit");
        }
        g.dispatch_at(dispatch_idx).workgroup_count_x = dx;
        g.dispatch_at(dispatch_idx).workgroup_count_y =
            static_cast<uint32_t>(dy64);
        g.set_cur_dims(
            out_id,
            {static_cast<int64_t>(live_b),
             static_cast<int64_t>(live_m),
             static_cast<int64_t>(live_n)});
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.bmm.default, bmm_impl);
}

} // namespace executorch::backends::webgpu
