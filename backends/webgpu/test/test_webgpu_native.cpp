/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDelegateHeader.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/rope/rotary_embedding_hf_wgsl.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

// Environment-derived config; captured in main() before RUN_ALL_TESTS().
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::string g_update_cache_model_path;
std::string g_qlinear_dir;
std::string g_prepack_model_path, g_prepack_golden_path;
std::string g_prepack2_model_path, g_prepack2_golden_path;
std::string g_prepack_tied_model_path, g_prepack_tied_golden_path;
std::string g_sdpa_dir;
std::string g_symint_blob;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

#ifdef WGPU_BACKEND_ENABLE_PROFILING
// Capacity-overrun must throw; runs without a device or TimestampQuery.
void test_query_pool_overrun_throws() {
  WebGPUQueryPool qp;
  EXPECT_THROW(qp.reset(1), std::exception)
      << "reset beyond capacity did not throw";
}

// WebGPUQueryPool roundtrip: time a probe pass; assert non-zero GPU duration.
void test_query_pool_roundtrip(const WebGPUContext& ctx) {
  if (!ctx.timestamp_supported) {
    GTEST_SKIP() << "adapter lacks TimestampQuery feature";
  }
  WGPUDevice device = ctx.device;

  // Probe loop iterates enough to burn a measurable, non-zero GPU duration.
  const char* kProbeWGSL =
      "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
      "  var acc = 0.0;\n"
      "  for (var i = 0u; i < 8192u; i = i + 1u) {\n"
      "    acc = acc + f32(i) * 1.000001;\n"
      "  }\n"
      "  out[gid.x] = acc;\n"
      "}\n";

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kProbeWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry bgl_entry = {};
  bgl_entry.binding = 0;
  bgl_entry.visibility = WGPUShaderStage_Compute;
  bgl_entry.buffer.type = WGPUBufferBindingType_Storage;
  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 1;
  bgl_desc.entries = &bgl_entry;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor pipe_desc = {};
  pipe_desc.layout = pl;
  pipe_desc.compute.module = shader;
  pipe_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipe =
      wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  WGPUBufferDescriptor obd = {};
  obd.size = 64 * sizeof(float);
  obd.usage = WGPUBufferUsage_Storage;
  WGPUBuffer out_buf = wgpuDeviceCreateBuffer(device, &obd);

  WGPUBindGroupEntry bg_entry = {};
  bg_entry.binding = 0;
  bg_entry.buffer = out_buf;
  bg_entry.size = obd.size;
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 1;
  bg_desc.entries = &bg_entry;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WebGPUQueryPool qp;
  qp.initialize(device, 1);
  qp.reset(1);

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
  WGPUPassTimestampWrites tw = qp.writes_for(0);
  WGPUComputePassDescriptor pass_desc = {};
  pass_desc.timestampWrites = &tw;
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(enc, &pass_desc);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  qp.record(0, "probe", {1, 1, 1}, {64, 1, 1});
  qp.resolve(enc);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(ctx.queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  qp.extract_results(ctx.instance);

  wgpuBufferRelease(out_buf);
  wgpuComputePipelineRelease(pipe);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuBindGroupRelease(bg);
  wgpuShaderModuleRelease(shader);

  ASSERT_EQ(qp.results().size(), 1u)
      << "expected 1 duration, got " << qp.results().size();
  const uint64_t dur = qp.results()[0].execution_duration_ns;
  printf("  probe duration: %llu ns\n", (unsigned long long)dur);
  EXPECT_NE(dur, 0u) << "probe duration is zero (expected monotonic non-zero)";
}

// Device-free: tick->duration delta math (the Mali begin-pinning fix).
void test_query_pool_delta_math() {
  auto durs = [](std::vector<uint32_t> idxs) {
    std::vector<ShaderDuration> v;
    for (uint32_t i : idxs) {
      ShaderDuration d;
      d.idx = i;
      v.push_back(d);
    }
    return v;
  };
  // Well-behaved backend (begin >= prev end): per-op == end - begin, unchanged.
  {
    const uint64_t ticks[] = {10, 20, 20, 35, 40, 50};
    auto d = durs({0, 1, 2});
    fill_shader_durations(d, ticks, 1.0);
    EXPECT_EQ(d[0].execution_duration_ns, 10u);
    EXPECT_EQ(d[1].execution_duration_ns, 15u);
    EXPECT_EQ(d[2].execution_duration_ns, 10u);
  }
  // Tile GPU: begin pinned, ends cumulative -> recover per-op; sum == wall.
  {
    const uint64_t ticks[] = {0, 20, 0, 50, 0, 90};
    auto d = durs({0, 1, 2});
    fill_shader_durations(d, ticks, 1.0);
    EXPECT_EQ(d[0].execution_duration_ns, 20u);
    EXPECT_EQ(d[1].execution_duration_ns, 30u);
    EXPECT_EQ(d[2].execution_duration_ns, 40u);
    const uint64_t sum = d[0].execution_duration_ns +
        d[1].execution_duration_ns + d[2].execution_duration_ns;
    EXPECT_EQ(sum, 90u);
  }
  // Recorded out of order: the idx-sort keeps the delta correct.
  {
    const uint64_t ticks[] = {0, 20, 0, 50, 0, 90};
    auto d = durs({2, 0, 1});
    fill_shader_durations(d, ticks, 1.0);
    for (const auto& x : d) {
      const uint64_t exp = x.idx == 0 ? 20u : (x.idx == 1 ? 30u : 40u);
      EXPECT_EQ(x.execution_duration_ns, exp) << "idx " << x.idx;
    }
  }
  // Non-monotone end (op1 end < prev end): running-max base + zero-clamp.
  {
    const uint64_t ticks[] = {0, 100, 0, 40, 0, 120};
    auto d = durs({0, 1, 2});
    fill_shader_durations(d, ticks, 1.0);
    EXPECT_EQ(d[0].start_time_ns, 0u);
    EXPECT_EQ(d[0].end_time_ns, 100u);
    EXPECT_EQ(d[0].execution_duration_ns, 100u);
    EXPECT_EQ(d[1].start_time_ns, 0u);
    EXPECT_EQ(d[1].end_time_ns, 40u);
    EXPECT_EQ(d[1].execution_duration_ns, 0u);
    EXPECT_EQ(d[2].start_time_ns, 0u);
    EXPECT_EQ(d[2].end_time_ns, 120u);
    EXPECT_EQ(d[2].execution_duration_ns, 20u);
  }
  // Each extraction is independent; no previous-end state leaks across calls.
  {
    const uint64_t first_ticks[] = {0, 100, 0, 140};
    auto first = durs({0, 1});
    fill_shader_durations(first, first_ticks, 1.0);
    EXPECT_EQ(first[1].execution_duration_ns, 40u);

    const uint64_t second_ticks[] = {10, 30, 30, 60};
    auto second = durs({0, 1});
    fill_shader_durations(second, second_ticks, 1.0);
    EXPECT_EQ(second[0].execution_duration_ns, 20u);
    EXPECT_EQ(second[1].execution_duration_ns, 30u);
  }
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

void test_update_cache(const std::string& model_path) {
  // update_cache: value [1,2,2,4] scattered into cache [1,8,2,4] at
  // input_pos=0.
  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int S = 2, H = 2, D = 4, Cmax = 8;
  constexpr int vnumel = S * H * D; // 16
  constexpr int cnumel = Cmax * H * D; // 64
  constexpr int input_pos = 0;

  std::vector<float> value(vnumel);
  std::vector<float> cache(cnumel);
  for (int i = 0; i < vnumel; i++) {
    value[i] = static_cast<float>(i) * 0.5f;
  }
  for (int i = 0; i < cnumel; i++) {
    cache[i] = static_cast<float>(i) + 100.0f;
  }

  // Reference: input_pos=0 overwrites the [0,S) seq slice of the cache with
  // value; the rest is preserved. Trivial scatter -- no library math involved.
  std::vector<float> ref(cache);
  for (int i = 0; i < vnumel; i++) {
    ref[input_pos * H * D + i] = value[i];
  }

  auto v = make_tensor_ptr({1, S, H, D}, std::vector<float>(value));
  auto c = make_tensor_ptr({1, Cmax, H, D}, std::vector<float>(cache));
  auto result = module.forward({EValue(v), EValue(c)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";

  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), cnumel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << cnumel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  for (int i = 0; i < cnumel; i++) {
    max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
  }
  printf("Max abs error: %e (checked %d elements)\n", max_abs_err, cnumel);
  EXPECT_LE(max_abs_err, 1e-3f) << "max error exceeds tolerance 1e-3";
}

std::vector<float> load_golden(const std::string& path, size_t numel) {
  // Load a raw little-endian fp32 golden written by the export .py (the native
  // binary has no ATen/torch, so the reference is computed offline).
  std::vector<float> g(numel);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  size_t n = std::fread(g.data(), sizeof(float), numel, f);
  std::fclose(f);
  if (n != numel) {
    return {};
  }
  return g;
}

// Per-element dual tolerance mirroring at::allclose's combined gate: an element
// is OK if within abs (1e-4) OR within rel (1e-3) tol, so a near-zero golden
// value can't blow up the rel metric (the kernel's ~1e-8 abs error is the real
// signal at llama3 scale). Sets the reported maxima; true iff all elements
// pass.
bool sdpa_within_tol(
    const float* out,
    const float* golden,
    int n,
    float* ma,
    float* mr,
    bool kv_f16 = false) {
  float atol = 1e-4f, rtol = 1e-3f;
  // Only fp16-KV cases receive the tolerance needed for storage rounding;
  // device capability alone must not weaken unrelated fp32 tests.
  if (kv_f16) {
    atol = 2e-3f;
    rtol = 1e-2f;
  }
  float max_abs = 0.0f, max_rel = 0.0f;
  bool ok = true;
  for (int i = 0; i < n; i++) {
    const float ae = std::abs(out[i] - golden[i]);
    const float re = ae / std::max(std::abs(golden[i]), 1e-6f);
    max_abs = std::max(max_abs, ae);
    max_rel = std::max(max_rel, re);
    if (ae > atol && re > rtol) {
      ok = false;
    }
  }
  *ma = max_abs;
  *mr = max_rel;
  return ok;
}

// Matches the WGSL Params struct in rms_norm.wgsl (16-byte aligned).
struct RmsNormProbeParams {
  uint32_t num_rows;
  uint32_t row_width;
  float epsilon;
  uint32_t _pad;
};

struct WgMapData {
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};
void wg_map_cb(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  static_cast<WgMapData*>(userdata1)->status = status;
}

// Run the rms_norm scalar kernel at an explicit override wg_size and map the
// output back. Module::forward can't set a second workgroup size (the handler
// always clamps to 64), so the pipeline is built directly here; the
// map/readback mirrors WebGPUGraph::copy_outputs.
std::vector<float> run_rms_norm_at_wg(
    const WebGPUContext& ctx,
    uint32_t wg_size,
    const std::vector<float>& input,
    const std::vector<float>& weight,
    uint32_t num_rows,
    uint32_t row_width,
    float epsilon) {
  WGPUDevice device = ctx.device;
  const uint64_t out_bytes =
      static_cast<uint64_t>(num_rows) * row_width * sizeof(float);
  const uint64_t in_bytes = static_cast<uint64_t>(input.size()) * sizeof(float);
  const uint64_t w_bytes = static_cast<uint64_t>(weight.size()) * sizeof(float);

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kRmsNormWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry bgl_entries[4] = {};
  bgl_entries[0].binding = 0;
  bgl_entries[0].visibility = WGPUShaderStage_Compute;
  bgl_entries[0].buffer.type = WGPUBufferBindingType_Storage;
  bgl_entries[1].binding = 1;
  bgl_entries[1].visibility = WGPUShaderStage_Compute;
  bgl_entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  bgl_entries[2].binding = 2;
  bgl_entries[2].visibility = WGPUShaderStage_Compute;
  bgl_entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  bgl_entries[3].binding = 3;
  bgl_entries[3].visibility = WGPUShaderStage_Compute;
  bgl_entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
  bgl_desc.entries = bgl_entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUConstantEntry wg_const = {};
  wg_const.key = {"wg_size", WGPU_STRLEN};
  wg_const.value = static_cast<double>(wg_size);

  WGPUComputePipelineDescriptor pipe_desc = {};
  pipe_desc.layout = pl;
  pipe_desc.compute.module = shader;
  pipe_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipe_desc.compute.constantCount = 1;
  pipe_desc.compute.constants = &wg_const;
  WGPUComputePipeline pipe =
      wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  WGPUBufferDescriptor out_bd = {};
  out_bd.size = out_bytes;
  out_bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
  WGPUBuffer out_buf = wgpuDeviceCreateBuffer(device, &out_bd);

  WGPUBufferDescriptor in_bd = {};
  in_bd.size = in_bytes;
  in_bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  WGPUBuffer in_buf = wgpuDeviceCreateBuffer(device, &in_bd);
  wgpuQueueWriteBuffer(ctx.queue, in_buf, 0, input.data(), in_bytes);

  WGPUBufferDescriptor w_bd = {};
  w_bd.size = w_bytes;
  w_bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  WGPUBuffer w_buf = wgpuDeviceCreateBuffer(device, &w_bd);
  wgpuQueueWriteBuffer(ctx.queue, w_buf, 0, weight.data(), w_bytes);

  RmsNormProbeParams params = {num_rows, row_width, epsilon, 0u};
  WGPUBufferDescriptor p_bd = {};
  p_bd.size = sizeof(RmsNormProbeParams);
  p_bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  WGPUBuffer p_buf = wgpuDeviceCreateBuffer(device, &p_bd);
  wgpuQueueWriteBuffer(ctx.queue, p_buf, 0, &params, sizeof(params));

  WGPUBufferDescriptor stg_bd = {};
  stg_bd.size = out_bytes;
  stg_bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &stg_bd);

  WGPUBindGroupEntry bg_entries[4] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out_buf;
  bg_entries[0].size = out_bytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = in_buf;
  bg_entries[1].size = in_bytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = w_buf;
  bg_entries[2].size = w_bytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = p_buf;
  bg_entries[3].size = sizeof(RmsNormProbeParams);
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
  WGPUComputePassDescriptor pass_desc = {};
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(enc, &pass_desc);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, num_rows, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  wgpuCommandEncoderCopyBufferToBuffer(enc, out_buf, 0, staging, 0, out_bytes);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(ctx.queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  WgMapData cb = {};
  WGPUBufferMapCallbackInfo cb_info = {};
  cb_info.mode = WGPUCallbackMode_WaitAnyOnly;
  cb_info.callback = wg_map_cb;
  cb_info.userdata1 = &cb;
  WGPUFuture fut =
      wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, out_bytes, cb_info);
  const WGPUWaitStatus wait = webgpu_wait(ctx.instance, fut);

  std::vector<float> result(static_cast<size_t>(num_rows) * row_width);
  bool ok = false;
  if (wait == WGPUWaitStatus_Success &&
      cb.status == WGPUMapAsyncStatus_Success) {
    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, out_bytes);
    std::memcpy(result.data(), mapped, out_bytes);
    wgpuBufferUnmap(staging);
    ok = true;
  }

  wgpuBufferRelease(staging);
  wgpuBufferRelease(p_buf);
  wgpuBufferRelease(w_buf);
  wgpuBufferRelease(in_buf);
  wgpuBufferRelease(out_buf);
  wgpuBindGroupRelease(bg);
  wgpuComputePipelineRelease(pipe);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuShaderModuleRelease(shader);

  if (!ok) {
    throw std::runtime_error("rms_norm wg-size probe: output map failed");
  }
  return result;
}

struct RotaryHfProbeParams {
  uint32_t n_heads;
  uint32_t seq;
  uint32_t head_dim;
  uint32_t half_dim;
  uint32_t num_pairs;
  uint32_t rotary_dim;
  uint32_t start_pos;
  uint32_t _pad;
};

std::vector<float> run_rope_hf_2d_probe(const WebGPUContext& ctx) {
  constexpr uint32_t kWorkgroupSize = 2;
  constexpr uint32_t kWorkgroupsX = 2;
  constexpr uint32_t kWorkgroupsY = 2;
  constexpr uint32_t kNumPairs = kWorkgroupSize * kWorkgroupsX * kWorkgroupsY;
  constexpr uint32_t kHeadDim = kNumPairs * 2;

  std::vector<float> input(kHeadDim);
  std::vector<float> output(kHeadDim, 0.0f);
  std::vector<float> freqs_cos(kHeadDim, 1.0f);
  std::vector<float> freqs_sin(kHeadDim, 0.0f);
  for (uint32_t i = 0; i < kHeadDim; i++) {
    input[i] = static_cast<float>(i + 1u);
    if (i >= kNumPairs) {
      freqs_cos[i] = 2.0f;
    }
  }

  WGPUDevice device = ctx.device;
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kRotaryEmbeddingHfWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUConstantEntry wg_const = {};
  wg_const.key = {"wg_size", WGPU_STRLEN};
  wg_const.value = static_cast<double>(kWorkgroupSize);
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_const;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
  WGPUBindGroupLayout layout =
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

  auto make_buffer =
      [device](const void* data, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc = {};
        desc.size = size;
        desc.usage = usage;
        desc.mappedAtCreation = true;
        WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
        std::memcpy(wgpuBufferGetMappedRange(buffer, 0, size), data, size);
        wgpuBufferUnmap(buffer);
        return buffer;
      };

  const uint64_t data_bytes = kHeadDim * sizeof(float);
  WGPUBuffer out_buffer = make_buffer(
      output.data(),
      data_bytes,
      WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
  WGPUBuffer in_buffer =
      make_buffer(input.data(), data_bytes, WGPUBufferUsage_Storage);
  WGPUBuffer cos_buffer =
      make_buffer(freqs_cos.data(), data_bytes, WGPUBufferUsage_Storage);
  WGPUBuffer sin_buffer =
      make_buffer(freqs_sin.data(), data_bytes, WGPUBufferUsage_Storage);
  const RotaryHfProbeParams params = {
      1u, 1u, kHeadDim, kNumPairs, kNumPairs, kHeadDim, 0u, 0u};
  WGPUBuffer params_buffer =
      make_buffer(&params, sizeof(params), WGPUBufferUsage_Uniform);

  WGPUBindGroupEntry entries[5] = {};
  const WGPUBuffer buffers[] = {
      out_buffer, in_buffer, cos_buffer, sin_buffer, params_buffer};
  const uint64_t sizes[] = {
      data_bytes, data_bytes, data_bytes, data_bytes, sizeof(params)};
  for (uint32_t i = 0; i < 5; i++) {
    entries[i].binding = i;
    entries[i].buffer = buffers[i];
    entries[i].size = sizes[i];
  }
  WGPUBindGroupDescriptor bind_group_desc = {};
  bind_group_desc.layout = layout;
  bind_group_desc.entryCount = 5;
  bind_group_desc.entries = entries;
  WGPUBindGroup bind_group =
      wgpuDeviceCreateBindGroup(device, &bind_group_desc);

  WGPUBufferDescriptor staging_desc = {};
  staging_desc.size = data_bytes;
  staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &staging_desc);

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
  WGPUComputePassDescriptor pass_desc = {};
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
  wgpuComputePassEncoderSetPipeline(pass, pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, kWorkgroupsX, kWorkgroupsY, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder, out_buffer, 0, staging, 0, data_bytes);
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, nullptr);
  wgpuQueueSubmit(ctx.queue, 1, &command);
  wgpuCommandBufferRelease(command);
  wgpuCommandEncoderRelease(encoder);

  WgMapData callback = {};
  WGPUBufferMapCallbackInfo callback_info = {};
  callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
  callback_info.callback = wg_map_cb;
  callback_info.userdata1 = &callback;
  WGPUFuture future = wgpuBufferMapAsync(
      staging, WGPUMapMode_Read, 0, data_bytes, callback_info);
  const WGPUWaitStatus wait = webgpu_wait(ctx.instance, future);
  if (wait == WGPUWaitStatus_Success &&
      callback.status == WGPUMapAsyncStatus_Success) {
    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, data_bytes);
    std::memcpy(output.data(), mapped, data_bytes);
    wgpuBufferUnmap(staging);
  } else {
    output.clear();
  }

  wgpuBufferRelease(staging);
  wgpuBindGroupRelease(bind_group);
  wgpuBufferRelease(params_buffer);
  wgpuBufferRelease(sin_buffer);
  wgpuBufferRelease(cos_buffer);
  wgpuBufferRelease(in_buffer);
  wgpuBufferRelease(out_buffer);
  wgpuBindGroupLayoutRelease(layout);
  wgpuComputePipelineRelease(pipeline);
  wgpuShaderModuleRelease(shader);
  return output;
}

// linear_q4gsw sweep config; mirrors CONFIGS in test_quantized_linear.py.
struct Q4gswConfig {
  const char* name;
  int m; // rows (tokens)
  int k; // in_features (reduction dim)
  int n; // out_features
  float tol_abs; // per-element abs gate
  float tol_rel; // per-element rel gate
  bool required; // dir set + .pte absent => FAIL (not skip)
  bool heavy; // huge/slow: export-gated; runs only if WEBGPU_TEST_HEAVY
};

// Llama-3.2-1B linear shapes (q/o/k/v/gate/up/down + lm_head) + 4k/8k prefill.
// tol scales with K (fp32 accum depth), not M; down_proj (K=8192) is looser.
const Q4gswConfig kQ4gswConfigs[] = {
    // name         M     K     N      tol_abs tol_rel req    heavy
    {"q_proj", 1, 2048, 2048, 1e-4f, 1e-3f, true, false},
    {"kv_proj", 1, 2048, 512, 1e-4f, 1e-3f, true, false},
    {"gate_proj", 1, 2048, 8192, 1e-4f, 1e-3f, true, false},
    {"down_proj", 1, 8192, 2048, 1e-3f, 1e-2f, true, false}, // big-K accum
    {"lm_head", 1, 2048, 128256, 1e-4f, 1e-3f, false, true},
    {"q_proj_4k", 4096, 2048, 2048, 1e-4f, 1e-3f, true, false},
    {"kv_proj_4k", 4096, 2048, 512, 1e-4f, 1e-3f, true, false},
    {"q_proj_8k", 8192, 2048, 2048, 1e-4f, 1e-3f, false, true},
    {"kv_proj_8k", 8192, 2048, 512, 1e-4f, 1e-3f, false, true},
    // The M==1 configs above (q/kv/gate/down_proj) exercise the bicol 2-col
    // decode GEMV (handler routes M==1 -> bicol; each reads its own per-column
    // scale over 64-256 K-groups). q4gsw requires N % 8 == 0, so odd-N is not
    // exportable; bicol's has1 odd-N guard is defensive.
    // M>1: steel GEMM on a >=256-invocation device (K%16==0), else shmem/tiled.
    {"steel", 96, 2048, 256, 1e-4f, 1e-3f, true, false}, // steel-isolating
    // Same shape as "steel" run under the f16-multiply steel kernel; the f16
    // rounding floor (~2.3e-4, uniform in K -- not an accumulate bug) needs a
    // looser abs gate than the strict f32 1e-4. Runs whenever the device
    // negotiated shader-f16 (else the f32 steel kernel; the looser gate holds).
    {"steel_f16", 96, 2048, 256, 2.3e-4f, 1e-3f, true, false},
    // Partial M and N steel tiles under the f16 kernel (f16 boundary masking).
    {"steel_f16_edge", 70, 1024, 136, 2.3e-4f, 1e-3f, true, false},
    // pwdq (packed-word dequant) backs the f16 steel path at group_size % BK ==
    // 0
    // (bit-exact to steel_half; the steel_f16 configs above run it at gs=32).
    // These lock the gs gate at group sizes those omit: gs=64 stays on pwdq;
    // gs=8 (< BK=16) falls back to the per-nibble steel_half kernel.
    {"pwdq_gs64", 96, 2048, 256, 2.3e-4f, 1e-3f, true, false},
    {"pwdq_gs8", 96, 2048, 256, 2.3e-4f, 1e-3f, true, false},
    // f16-ACCUMULATE steel (pwdqf16acc): lossy, so a wider gate than the
    // f16-multiply steel_f16 (2.3e-4). f16 accumulation error grows with K, so
    // the deep-K down shape (K=8192) gets the loosest tol. Perplexity is the
    // primary quality gate (see the kernel diff); this catches gross bit/index
    // bugs. gs=32 (% BK == 0) selects pwdqf16acc; the sweep loads these rows
    // with the enable_f16_accumulate_gemm runtime spec set.
    {"pwdqf16acc", 96, 2048, 256, 2e-2f, 3e-2f, true, false},
    {"pwdqf16acc_down", 128, 8192, 2048, 5e-2f, 8e-2f, true, false},
    {"gate_proj_pf", 128, 2048, 8192, 1e-4f, 1e-3f, true, false}, // shmem via N
    {"down_proj_pf", 128, 8192, 2048, 1e-3f, 1e-2f, true, false}, // shmem via K
    {"shmem_edge", 130, 4096, 2056, 1e-4f, 1e-3f, true, false}, // partial tiles
};

// /16 ramp over the flat index; mirrors test_quantized_linear.py _ramp_input.
float q4gsw_ramp(int i) {
  return static_cast<float>((i % 17) - 8) / 16.0f;
}

// Per-element abs-OR-rel tolerance helper.
bool quant_within_tol(
    const float* out,
    const float* golden,
    int n,
    float atol,
    float rtol,
    float* ma,
    float* mr) {
  float max_abs = 0.0f, max_rel = 0.0f;
  bool ok = true;
  for (int i = 0; i < n; i++) {
    const float ae = std::abs(out[i] - golden[i]);
    const float re = ae / std::max(std::abs(golden[i]), 1e-6f);
    max_abs = std::max(max_abs, ae);
    max_rel = std::max(max_rel, re);
    if (ae > atol && re > rtol) {
      ok = false;
    }
  }
  *ma = max_abs;
  *mr = max_rel;
  return ok;
}

std::vector<int32_t> load_indices(const std::string& path, size_t numel) {
  // Load raw little-endian int32 indices written by the export .py.
  std::vector<int32_t> g(numel);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  size_t n = std::fread(g.data(), sizeof(int32_t), numel, f);
  std::fclose(f);
  if (n != numel) {
    return {};
  }
  return g;
}

void test_embedding_q4gsw(
    const std::string& model_path,
    const std::string& indices_path,
    const std::string& golden_path,
    int num_indices,
    int embed,
    const char* label) {
  // q4gsw embedding-gather vs torch golden; shapes per test_embedding_q4gsw.py.
  const int out_numel = num_indices * embed;
  printf(
      "\n--- Test: embedding_q4gsw (%s: indices=%d, embed=%d) ---\n",
      label,
      num_indices,
      embed);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  std::vector<int32_t> idx32 = load_indices(indices_path, num_indices);
  std::vector<float> golden = load_golden(golden_path, out_numel);
  ASSERT_FALSE(idx32.empty() || golden.empty())
      << "could not load indices " << indices_path << " / golden "
      << golden_path;

  // int64 at the program boundary; copy_inputs narrows to the int32 buffer.
  std::vector<int64_t> idx64(idx32.begin(), idx32.end());
  auto idx = make_tensor_ptr({num_indices}, std::move(idx64));

  auto result = module.forward({EValue(idx)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), out_numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << out_numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  const bool pass = quant_within_tol(
      out_data,
      golden.data(),
      out_numel,
      1e-3f,
      1e-3f,
      &max_abs_err,
      &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      out_numel);
  EXPECT_TRUE(pass) << "embedding_q4gsw exceeds tolerance 1e-3 (abs AND rel)";
}

void test_rope(
    const std::string& model_path,
    const std::string& xq_golden_path,
    const std::string& xk_golden_path,
    int S,
    int NH,
    int NKV,
    int HD,
    const char* label) {
  // Llama interleaved RoPE vs torch goldens; shapes/ramps per test_rope.py.
  const int xq_numel = S * NH * HD;
  const int xk_numel = S * NKV * HD;
  const int freqs_numel = S * (HD / 2);
  printf(
      "\n--- Test: apply_rotary_emb (%s: S=%d,NH=%d,NKV=%d,HD=%d) ---\n",
      label,
      S,
      NH,
      NKV,
      HD);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  // ((i % mod) - off) / 16: exact in fp32, matches test_rope.py::_ramp.
  auto ramp = [](int i, int mod, int off) {
    return static_cast<float>((i % mod) - off) / 16.0f;
  };
  std::vector<float> xq(xq_numel), xk(xk_numel), fc(freqs_numel),
      fs(freqs_numel);
  for (int i = 0; i < xq_numel; i++) {
    xq[i] = ramp(i, 17, 8);
  }
  for (int i = 0; i < xk_numel; i++) {
    xk[i] = ramp(i, 13, 6);
  }
  for (int i = 0; i < freqs_numel; i++) {
    fc[i] = ramp(i, 11, 5);
    fs[i] = ramp(i, 7, 3);
  }

  auto xqt = make_tensor_ptr({1, S, NH, HD}, std::vector<float>(xq));
  auto xkt = make_tensor_ptr({1, S, NKV, HD}, std::vector<float>(xk));
  auto fct = make_tensor_ptr({S, HD / 2}, std::vector<float>(fc));
  auto fst = make_tensor_ptr({S, HD / 2}, std::vector<float>(fs));

  auto result =
      module.forward({EValue(xqt), EValue(xkt), EValue(fct), EValue(fst)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();

  // Outputs in graph order [0]=xq_out, [1]=xk_out (positional; the numel check
  // below guards a swap, since NH != NKV under GQA).
  ASSERT_TRUE(
      outputs.size() >= 2 && outputs[0].isTensor() && outputs[1].isTensor())
      << "expected 2 tensor outputs, got " << outputs.size();
  const auto& xq_t = outputs[0].toTensor();
  const auto& xk_t = outputs[1].toTensor();
  ASSERT_TRUE(xq_t.numel() == xq_numel && xk_t.numel() == xk_numel)
      << "output shapes [" << (size_t)xq_t.numel() << ","
      << (size_t)xk_t.numel() << "] != expected [" << xq_numel << ","
      << xk_numel << "]";
  const float* xq_out = xq_t.const_data_ptr<float>();
  const float* xk_out = xk_t.const_data_ptr<float>();

  std::vector<float> gq = load_golden(xq_golden_path, xq_numel);
  std::vector<float> gk = load_golden(xk_golden_path, xk_numel);
  ASSERT_FALSE(gq.empty() || gk.empty())
      << "could not load goldens " << xq_golden_path << " / " << xk_golden_path;

  // Per-element abs-OR-rel on xq and xk (shared helper).
  float maq = 0.0f, mrq = 0.0f, mak = 0.0f, mrk = 0.0f;
  const bool pass_q =
      quant_within_tol(xq_out, gq.data(), xq_numel, 1e-3f, 1e-3f, &maq, &mrq);
  const bool pass_k =
      quant_within_tol(xk_out, gk.data(), xk_numel, 1e-3f, 1e-3f, &mak, &mrk);
  const float max_abs_err = std::max(maq, mak);
  const float max_rel_err = std::max(mrq, mrk);

  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      xq_numel + xk_numel);
  EXPECT_TRUE(pass_q && pass_k)
      << "apply_rotary_emb exceeds tolerance 1e-3 (abs AND rel)";
}

bool has_shape(
    const executorch::aten::Tensor& tensor,
    const std::vector<int64_t>& expected) {
  if (tensor.dim() != static_cast<int64_t>(expected.size())) {
    return false;
  }
  for (size_t i = 0; i < expected.size(); i++) {
    if (tensor.size(static_cast<int64_t>(i)) != expected[i]) {
      return false;
    }
  }
  return true;
}

void test_rope_hf_dynamic(const std::string& dir) {
  constexpr int S = 1;
  constexpr int NH = 16;
  constexpr int NKV = 8;
  constexpr int HD = 128;
  constexpr int MAXS = 16;
  constexpr int positions[] = {0, 7, 15};
  constexpr int xq_numel = S * NH * HD;
  constexpr int xk_numel = S * NKV * HD;
  constexpr int freqs_numel = MAXS * HD;

  Module module(dir + "rope_hf_dynamic.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load HF RoPE dynamic model";

  std::vector<float> xq = load_golden(dir + "rope_hf_dynamic.xq.bin", xq_numel);
  std::vector<float> xk = load_golden(dir + "rope_hf_dynamic.xk.bin", xk_numel);
  std::vector<float> freqs_cos =
      load_golden(dir + "rope_hf_dynamic.freqs_cos.bin", freqs_numel);
  std::vector<float> freqs_sin =
      load_golden(dir + "rope_hf_dynamic.freqs_sin.bin", freqs_numel);
  ASSERT_FALSE(
      xq.empty() || xk.empty() || freqs_cos.empty() || freqs_sin.empty())
      << "could not load HF RoPE input binaries from " << dir;

  for (const int position : positions) {
    auto xqt = make_tensor_ptr({1, S, NH, HD}, std::vector<float>(xq));
    auto xkt = make_tensor_ptr({1, S, NKV, HD}, std::vector<float>(xk));
    auto fct = make_tensor_ptr({MAXS, HD}, std::vector<float>(freqs_cos));
    auto fst = make_tensor_ptr({MAXS, HD}, std::vector<float>(freqs_sin));
    auto post = make_tensor_ptr(
        {1}, std::vector<int64_t>{static_cast<int64_t>(position)});
    auto result = module.forward(
        {EValue(xqt), EValue(xkt), EValue(fct), EValue(fst), EValue(post)});
    ASSERT_TRUE(result.ok())
        << "HF RoPE forward failed at position " << position << " (error "
        << static_cast<int>(result.error()) << ")";
    const auto& outputs = result.get();
    ASSERT_TRUE(
        outputs.size() == 2 && outputs[0].isTensor() && outputs[1].isTensor())
        << "expected exactly two HF RoPE tensor outputs";
    const auto& xq_out = outputs[0].toTensor();
    const auto& xk_out = outputs[1].toTensor();
    ASSERT_TRUE(has_shape(xq_out, {1, S, NH, HD}))
        << "HF RoPE query output has the wrong shape at position " << position;
    ASSERT_TRUE(has_shape(xk_out, {1, S, NKV, HD}))
        << "HF RoPE key output has the wrong shape at position " << position;

    const std::string prefix =
        dir + "rope_hf_dynamic.pos" + std::to_string(position);
    const std::vector<float> golden_q =
        load_golden(prefix + ".xq.golden.bin", xq_numel);
    const std::vector<float> golden_k =
        load_golden(prefix + ".xk.golden.bin", xk_numel);
    ASSERT_FALSE(golden_q.empty() || golden_k.empty())
        << "could not load HF RoPE goldens for position " << position;

    float q_abs = 0.0f, q_rel = 0.0f, k_abs = 0.0f, k_rel = 0.0f;
    const bool q_ok = quant_within_tol(
        xq_out.const_data_ptr<float>(),
        golden_q.data(),
        xq_numel,
        1e-4f,
        1e-3f,
        &q_abs,
        &q_rel);
    const bool k_ok = quant_within_tol(
        xk_out.const_data_ptr<float>(),
        golden_k.data(),
        xk_numel,
        1e-4f,
        1e-3f,
        &k_abs,
        &k_rel);
    EXPECT_TRUE(q_ok && k_ok)
        << "HF RoPE mismatch at position " << position << ": q abs=" << q_abs
        << " rel=" << q_rel << ", k abs=" << k_abs << " rel=" << k_rel;
  }

  auto xqt = make_tensor_ptr({1, S, NH, HD}, std::vector<float>(xq));
  auto xkt = make_tensor_ptr({1, S, NKV, HD}, std::vector<float>(xk));
  auto fct = make_tensor_ptr({MAXS, HD}, std::move(freqs_cos));
  auto fst = make_tensor_ptr({MAXS, HD}, std::move(freqs_sin));

  auto overflow_post =
      make_tensor_ptr({1}, std::vector<int64_t>{INT64_C(1) << 32});
  auto overflow = module.forward({
      EValue(xqt),
      EValue(xkt),
      EValue(fct),
      EValue(fst),
      EValue(overflow_post),
  });
  EXPECT_FALSE(overflow.ok())
      << "HF RoPE accepted a start_pos that aliases to zero when narrowed";

  auto post = make_tensor_ptr({1}, std::vector<int64_t>{MAXS});
  auto out_of_range = module.forward(
      {EValue(xqt), EValue(xkt), EValue(fct), EValue(fst), EValue(post)});
  EXPECT_FALSE(out_of_range.ok())
      << "HF RoPE accepted start_pos + seq beyond the frequency table";

  auto negative_post = make_tensor_ptr({1}, std::vector<int64_t>{-1});
  auto negative = module.forward({
      EValue(xqt),
      EValue(xkt),
      EValue(fct),
      EValue(fst),
      EValue(negative_post),
  });
  EXPECT_FALSE(negative.ok()) << "HF RoPE accepted a negative start_pos";
}

void test_rope_hf_dynamic_sequence_reused_graph(const std::string& dir) {
  constexpr int NH = 16;
  constexpr int NKV = 8;
  constexpr int HD = 128;
  constexpr int MAXS = 16;
  struct Case {
    int seq;
    int position;
  };
  constexpr Case cases[] = {{16, 0}, {5, 7}, {1, 15}, {16, 0}};

  const std::string prefix = dir + "rope_hf_dynamic_sequence";
  Module module(prefix + ".pte");
  ASSERT_EQ(module.load_forward(), Error::Ok)
      << "could not load HF RoPE dynamic-sequence model";

  const int freqs_numel = MAXS * HD;
  const std::vector<float> freqs_cos =
      load_golden(prefix + ".freqs_cos.bin", freqs_numel);
  const std::vector<float> freqs_sin =
      load_golden(prefix + ".freqs_sin.bin", freqs_numel);
  ASSERT_FALSE(freqs_cos.empty() || freqs_sin.empty())
      << "could not load HF RoPE dynamic-sequence frequencies";

  for (const Case& c : cases) {
    const int xq_numel = c.seq * NH * HD;
    const int xk_numel = c.seq * NKV * HD;
    const std::string case_prefix = prefix + ".S" + std::to_string(c.seq) +
        ".pos" + std::to_string(c.position);
    const std::vector<float> xq =
        load_golden(case_prefix + ".xq.bin", xq_numel);
    const std::vector<float> xk =
        load_golden(case_prefix + ".xk.bin", xk_numel);
    const std::vector<float> golden_q =
        load_golden(case_prefix + ".xq.golden.bin", xq_numel);
    const std::vector<float> golden_k =
        load_golden(case_prefix + ".xk.golden.bin", xk_numel);
    ASSERT_FALSE(
        xq.empty() || xk.empty() || golden_q.empty() || golden_k.empty())
        << "could not load HF RoPE dynamic-sequence case " << case_prefix;

    auto xqt = make_tensor_ptr({1, c.seq, NH, HD}, std::vector<float>(xq));
    auto xkt = make_tensor_ptr({1, c.seq, NKV, HD}, std::vector<float>(xk));
    auto fct = make_tensor_ptr({MAXS, HD}, std::vector<float>(freqs_cos));
    auto fst = make_tensor_ptr({MAXS, HD}, std::vector<float>(freqs_sin));
    auto post = make_tensor_ptr(
        {1}, std::vector<int64_t>{static_cast<int64_t>(c.position)});
    auto result = module.forward(
        {EValue(xqt), EValue(xkt), EValue(fct), EValue(fst), EValue(post)});
    ASSERT_TRUE(result.ok())
        << "HF RoPE dynamic-sequence forward failed for " << case_prefix
        << " (error " << static_cast<int>(result.error()) << ")";
    const auto& outputs = result.get();
    ASSERT_TRUE(
        outputs.size() == 2 && outputs[0].isTensor() && outputs[1].isTensor())
        << "expected exactly two HF RoPE dynamic-sequence tensor outputs";
    const auto& xq_out = outputs[0].toTensor();
    const auto& xk_out = outputs[1].toTensor();
    ASSERT_TRUE(has_shape(xq_out, {1, c.seq, NH, HD}))
        << "HF RoPE query output has the wrong shape for " << case_prefix;
    ASSERT_TRUE(has_shape(xk_out, {1, c.seq, NKV, HD}))
        << "HF RoPE key output has the wrong shape for " << case_prefix;

    float q_abs = 0.0f, q_rel = 0.0f, k_abs = 0.0f, k_rel = 0.0f;
    const bool q_ok = quant_within_tol(
        xq_out.const_data_ptr<float>(),
        golden_q.data(),
        xq_numel,
        1e-4f,
        1e-3f,
        &q_abs,
        &q_rel);
    const bool k_ok = quant_within_tol(
        xk_out.const_data_ptr<float>(),
        golden_k.data(),
        xk_numel,
        1e-4f,
        1e-3f,
        &k_abs,
        &k_rel);
    EXPECT_TRUE(q_ok && k_ok)
        << "HF RoPE dynamic-sequence mismatch for " << case_prefix
        << ": q abs=" << q_abs << " rel=" << q_rel << ", k abs=" << k_abs
        << " rel=" << k_rel;
  }
}

void test_prepack(
    const std::string& model_path,
    const std::string& golden_path,
    const std::string& label = "x + const w") {
  // et_vk.prepack copy vs golden; unrun copy leaves zeros. See test_prepack.py.
  constexpr int n = 4;
  constexpr int numel = n * n;
  printf("\n--- Test: prepack (%s, %dx%d) ---\n", label.c_str(), n, n);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  std::vector<float> golden = load_golden(golden_path, numel);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  // ((i % 13) - 6) / 16: exact in fp32, matches test_prepack.py::_inputs.
  std::vector<float> x_data(numel);
  for (int i = 0; i < numel; i++) {
    x_data[i] = static_cast<float>((i % 13) - 6) / 16.0f;
  }
  auto x = make_tensor_ptr({n, n}, std::vector<float>(x_data));

  auto result = module.forward({EValue(x)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  // Per-element abs-OR-rel (quant_within_tol): a global rel gate spuriously
  // fails near-zero outputs where rel error explodes.
  const bool within = quant_within_tol(
      out_data, golden.data(), numel, 1e-3f, 1e-3f, &max_abs_err, &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      numel);
  EXPECT_TRUE(within) << "prepack exceeds tolerance 1e-3";
}

// Reconstruct _ramp_input bit-for-bit, run the op, compare to the fp64 golden.
void test_q4gsw_config(
    const Q4gswConfig& cfg,
    const std::string& pte,
    const std::string& golden_path) {
  printf(
      "\n--- Test: linear_q4gsw (%s: M=%d,K=%d,N=%d) ---\n",
      cfg.name,
      cfg.m,
      cfg.k,
      cfg.n);

  Module module(pte);
  // pwdqf16acc rows exercise the lossy f16-accumulate kernel, a runtime opt-in
  // (default off); enable it via the backend option keyed by the registered id.
  if (std::string(cfg.name).rfind("pwdqf16acc", 0) == 0) {
    BackendOptions<1> opts;
    opts.set_option("enable_f16_accumulate_gemm", true);
    LoadBackendOptionsMap map;
    ASSERT_EQ(map.set_options("VulkanBackend", opts.view()), Error::Ok);
    ASSERT_EQ(module.load_forward(nullptr, nullptr, &map), Error::Ok)
        << "could not load " << pte;
  } else {
    ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load " << pte;
  }

  const int in_numel = cfg.m * cfg.k;
  const int out_numel = cfg.m * cfg.n;
  std::vector<float> input(in_numel);
  for (int i = 0; i < in_numel; i++) {
    input[i] = q4gsw_ramp(i);
  }

  auto x = make_tensor_ptr({cfg.m, cfg.k}, std::vector<float>(input));
  auto result = module.forward({EValue(x)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), out_numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << out_numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  std::vector<float> golden = load_golden(golden_path, out_numel);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  float ma = 0.0f, mr = 0.0f;
  const bool pass = quant_within_tol(
      out_data, golden.data(), out_numel, cfg.tol_abs, cfg.tol_rel, &ma, &mr);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      ma,
      mr,
      out_numel);
  EXPECT_TRUE(pass) << "linear_q4gsw " << cfg.name << " exceeds tolerance (abs "
                    << cfg.tol_abs << " OR rel " << cfg.tol_rel << ")";
}

// Fused sdpa_with_kv_cache sweep config. Mirrors the Python CONFIGS table in
// test_sdpa.py exactly (name, Hq, Hkv, D, S, Cmax, input_pos).
struct SdpaConfig {
  const char* name;
  int hq; // query heads
  int hkv; // key/value heads (GQA groups when hq != hkv)
  int d; // head dim
  int s; // new tokens this step
  int cmax; // kv-cache capacity
  int input_pos; // prior tokens already in the cache (decode)
  float denom; // ramp divisor (mirrors Python); small -> large logits
  bool required = false; // CI (SDPA dir set): absent .pte = FAIL, not skip
  bool expect_reject = false; // load MUST fail (e.g. D%4 guard), no golden
  bool kv_f16 = false;
};

const SdpaConfig kSdpaConfigs[] = {
    // name             Hq Hkv  D  S Cmax pos denom
    {"gqa31_prefill", 6, 2, 8, 4, 16, 0, 16.0f}, // GQA 3:1 (original case)
    {"mha_ctxodd", 4, 4, 16, 3, 8, 0, 16.0f}, // MHA; context_len=3 (odd)
    {"gqa21_prefill", 8, 4, 4, 5, 16, 0, 16.0f}, // GQA 2:1; multi-token S=5
    {"gqa31_decode", 6, 2, 8, 2, 16, 2, 16.0f}, // decode: 2 prior tokens
    // llama3-ish GQA, D=128, S=128.
    {"llama3_prefill", 24, 8, 128, 128, 256, 0, 16.0f},
    // Adversarial: denom=0.5 -> peak logit ~177 (>88) overflows naive fp32 exp.
    {"mha_biglogit", 4, 4, 32, 4, 16, 0, 0.5f},
    // Llama 3.2 1B shape (Hq=32,Hkv=8,D=64): decode at 4k/8k ctx.
    {"llama1b_decode_4k", 32, 8, 64, 1, 4096, 4095, 16.0f, /*required=*/true},
    {"llama1b_decode_8k", 32, 8, 64, 1, 8192, 8191, 16.0f, /*required=*/true},
    // Llama 3.2 1B shape: realistic prefill (S=128 at pos 0) + decode (S=1 at
    // pos 127).
    {"llama1b_prefill", 32, 8, 64, 128, 512, 0, 16.0f},
    {"llama1b_decode", 32, 8, 64, 1, 512, 127, 16.0f},
    // D=6 is not a multiple of 4: the head_dim%4 guard must reject it at load.
    {"reject_d6",
     4,
     4,
     6,
     4,
     16,
     0,
     16.0f,
     /*required=*/false,
     /*expect_reject=*/true},
    // 2D-dispatch cap (>65535 wg): S=512 folds QK; S=2048 folds QK+softmax+AV
    // (cap+1).
    {"llama1b_prefill_512", 32, 8, 64, 512, 512, 0, 16.0f, /*required=*/true},
    {"llama1b_prefill_2048",
     32,
     8,
     64,
     2048,
     2048,
     0,
     16.0f,
     /*required=*/true},
    {"qwen3_prefill",
     16,
     8,
     128,
     128,
     256,
     0,
     10.0f,
     /*required=*/true,
     /*expect_reject=*/false,
     /*kv_f16=*/true},
    {"qwen3_odd_boundary",
     16,
     8,
     128,
     17,
     64,
     31,
     10.0f,
     /*required=*/true,
     /*expect_reject=*/false,
     /*kv_f16=*/true},
};

// Ramp denominator; mirror of test_sdpa.py::_RAMP_DENOM (keep in sync).
constexpr float kSdpaRampDenom = 16.0f;

// /denom ramp: ((i % mod) - off) / denom, exact in fp32 (power-of-two denom).
// Mirrors test_sdpa.py::_ramp.
float sdpa_ramp(int i, int mod, int off, float denom = kSdpaRampDenom) {
  return static_cast<float>((i % mod) - off) / denom;
}

// Step-indexed ramp; mirrors test_sdpa.py::_ramp_t bit-for-bit. denom defaults
// to kSdpaRampDenom and must match the Python denom for bit-identity.
float sdpa_ramp_t(
    int i,
    int mod,
    int off,
    int t,
    float denom = kSdpaRampDenom) {
  return static_cast<float>(((i + 31 * t) % mod) - off) / denom;
}

// Multi-step replay sequences. The first three mirror Vulkan param sets; Qwen3
// extends the same Python REPLAY_SEQS contract.
struct SdpaSequence {
  const char* name;
  int hq;
  int hkv;
  int d;
  int cmax;
  std::vector<int> seq_lens;
  bool kv_f16 = false;
};

const SdpaSequence kSdpaSequences[] = {
    {"small", 8, 4, 4, 16, {3, 1, 1, 5, 1, 1, 2}},
    {"small_d", 6, 2, 8, 16, {3, 1, 1, 5, 1, 1}},
    {"llama3", 24, 8, 128, 256, {111, 1, 1, 1, 57, 1, 1}},
    {"qwen3_fd", 16, 8, 128, 64, {17, 1}, /*kv_f16=*/true},
};

Error load_sdpa_forward(Module& module, bool kv_f16, int sdpa_query_tile = 0) {
  if (!kv_f16 && sdpa_query_tile == 0) {
    return module.load_forward();
  }
  BackendOptions<2> options;
  Error error = Error::Ok;
  if (kv_f16) {
    error = options.set_option("enable_f16_kv_cache", true);
    if (error != Error::Ok) {
      return error;
    }
  }
  if (sdpa_query_tile != 0) {
    error = options.set_option("sdpa_query_tile", sdpa_query_tile);
    if (error != Error::Ok) {
      return error;
    }
  }
  LoadBackendOptionsMap option_map;
  error = option_map.set_options("VulkanBackend", options.view());
  if (error != Error::Ok) {
    return error;
  }
  return module.load_forward(nullptr, nullptr, &option_map);
}

bool shader_f16_supported_on_test_device() {
  const WebGPUContext* context = get_default_webgpu_context();
  return context != nullptr && context->shader_f16_supported;
}

bool qwen3_q16_supported_on_test_device() {
  constexpr uint32_t kQ16StorageBytes = 512u * 4u * sizeof(float) +
      512u * 4u * sizeof(uint16_t) + 128u * 2u * sizeof(float) +
      3u * 16u * sizeof(float);
  const WebGPUContext* context = get_default_webgpu_context();
  WGPULimits limits = {};
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(context->device, &limits) == WGPUStatus_Success &&
      limits.maxComputeWorkgroupSizeX >= 16u &&
      limits.maxComputeWorkgroupSizeY >= 8u &&
      limits.maxComputeInvocationsPerWorkgroup >= 128u &&
      limits.maxComputeWorkgroupStorageSize >= kQ16StorageBytes &&
      limits.maxStorageBuffersPerShaderStage >= 4u;
}

bool qwen3_q32_supported_on_test_device() {
  constexpr uint32_t kQ32StorageBytes = 1024u * 4u * sizeof(float) +
      512u * 4u * sizeof(uint16_t) + 256u * 2u * sizeof(float) +
      3u * 32u * sizeof(float);
  const WebGPUContext* context = get_default_webgpu_context();
  WGPULimits limits = {};
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(context->device, &limits) == WGPUStatus_Success &&
      limits.maxComputeWorkgroupSizeX >= 32u &&
      limits.maxComputeWorkgroupSizeY >= 8u &&
      limits.maxComputeInvocationsPerWorkgroup >= 256u &&
      limits.maxComputeWorkgroupStorageSize >= kQ32StorageBytes &&
      limits.maxStorageBuffersPerShaderStage >= 4u;
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
constexpr uint32_t kTestRouteMaterializedAttention = 1u << 2;
constexpr uint32_t kTestRouteFlashDecoding = 1u << 10;
constexpr uint32_t kTestRouteK16CausalBound = 1u << 11;
constexpr uint32_t kTestRouteQwen3Q16K16 = 1u << 13;
constexpr uint32_t kTestRouteQwen3Q32K16 = 1u << 14;
#endif // WGPU_BACKEND_ENABLE_PROFILING

void test_sdpa_config(
    const SdpaConfig& cfg,
    const std::string& model_path,
    const std::string& golden_path,
    int sdpa_query_tile = 0) {
  // Inputs reconstruct test_sdpa.py::_det_inputs bit-for-bit (/16 exact fp32).
  printf(
      "\n--- Test: sdpa_with_kv_cache (%s: Hq=%d,Hkv=%d,D=%d,S=%d,Cmax=%d,pos=%d) ---\n",
      cfg.name,
      cfg.hq,
      cfg.hkv,
      cfg.d,
      cfg.s,
      cfg.cmax,
      cfg.input_pos);

  if (cfg.kv_f16 && !shader_f16_supported_on_test_device()) {
    printf("SKIP: %s requires shader-f16\n", cfg.name);
    return;
  }

  Module module(model_path);
  auto err = load_sdpa_forward(module, cfg.kv_f16, sdpa_query_tile);
  if (cfg.expect_reject) {
    // D not a multiple of 4 must be rejected at load by the head_dim guard.
    ASSERT_NE(err, Error::Ok)
        << cfg.name << " loaded OK; head_dim%4 guard did not fire";
    printf("PASS: %s rejected at load (error %d)\n", cfg.name, (int)err);
    return;
  }
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  const int qn = cfg.s * cfg.hq * cfg.d;
  const int kn = cfg.s * cfg.hkv * cfg.d;
  const int cn = cfg.cmax * cfg.hkv * cfg.d;
  const int on = cfg.s * cfg.hq * cfg.d;

  std::vector<float> q(qn), k(kn), v(kn), kc(cn, 0.0f), vc(cn, 0.0f);
  for (int i = 0; i < qn; i++) {
    q[i] = sdpa_ramp(i, 17, 8, cfg.denom);
  }
  for (int i = 0; i < kn; i++) {
    k[i] = sdpa_ramp(i, 13, 6, cfg.denom);
    v[i] = sdpa_ramp(i, 11, 5, cfg.denom);
  }
  // Decode: seed cache rows [0, input_pos) with prior_k/prior_v (flat over
  // input_pos*Hkv*D elements); all other rows stay zero.
  const int prior_n = cfg.input_pos * cfg.hkv * cfg.d;
  for (int i = 0; i < prior_n; i++) {
    kc[i] = sdpa_ramp(i, 7, 3);
    vc[i] = sdpa_ramp(i, 5, 2);
  }

  auto qt = make_tensor_ptr({1, cfg.s, cfg.hq, cfg.d}, std::vector<float>(q));
  auto kt = make_tensor_ptr({1, cfg.s, cfg.hkv, cfg.d}, std::vector<float>(k));
  auto vt = make_tensor_ptr({1, cfg.s, cfg.hkv, cfg.d}, std::vector<float>(v));
  auto kct =
      make_tensor_ptr({1, cfg.cmax, cfg.hkv, cfg.d}, std::vector<float>(kc));
  auto vct =
      make_tensor_ptr({1, cfg.cmax, cfg.hkv, cfg.d}, std::vector<float>(vc));

  auto result = module.forward(
      {EValue(qt), EValue(kt), EValue(vt), EValue(kct), EValue(vct)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  if (cfg.kv_f16) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    // Exact Qwen3 geometry + fp16 KV selects the K16 streaming (causal-bound)
    // route by default. The sdpa_query_tile RuntimeSpec only swaps the Q16/Q32
    // kernel variant; both map to the K16CausalBound bit. A non-Qwen3 fp16-KV
    // shape falls back to the materialized path (or flash-decoding at S==1).
    const bool qwen3_geometry = cfg.hq == 16 && cfg.hkv == 8 && cfg.d == 128;
    const bool qwen3_streaming =
        qwen3_geometry && cfg.s > 1 && qwen3_q16_supported_on_test_device();
    const uint32_t expected_route = qwen3_streaming
        ? kTestRouteK16CausalBound
        : (cfg.s == 1 ? kTestRouteFlashDecoding
                      : kTestRouteMaterializedAttention);
    EXPECT_EQ(
        g_last_route_mask &
            (kTestRouteMaterializedAttention | kTestRouteFlashDecoding |
             kTestRouteK16CausalBound),
        expected_route);
    EXPECT_EQ(g_last_route_conflict_count, 0u);
    const uint32_t qwen3_tile_routes =
        g_last_route_mask & (kTestRouteQwen3Q16K16 | kTestRouteQwen3Q32K16);
    if (qwen3_streaming) {
      const uint32_t expected_tile_route =
          sdpa_query_tile == 32 && qwen3_q32_supported_on_test_device()
          ? kTestRouteQwen3Q32K16
          : kTestRouteQwen3Q16K16;
      EXPECT_EQ(qwen3_tile_routes, expected_tile_route);
    } else {
      EXPECT_EQ(qwen3_tile_routes, 0u);
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING
  }

  const auto& outputs = result.get();
  // Select the attention output [1,S,Hq,D] by shape; the op returns
  // [k_cache, v_cache, attn_output] and a cache [1,Cmax,Hkv,D] can share numel.
  int attn_idx = -1;
  int attn_matches = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (!outputs[i].isTensor()) {
      continue;
    }
    const auto& t = outputs[i].toTensor();
    if (t.dim() == 4 && static_cast<int>(t.size(1)) == cfg.s &&
        static_cast<int>(t.size(2)) == cfg.hq &&
        static_cast<int>(t.size(3)) == cfg.d) {
      attn_idx = static_cast<int>(i);
      attn_matches++;
    }
  }
  ASSERT_GE(attn_idx, 0) << "no attention output [1," << cfg.s << "," << cfg.hq
                         << "," << cfg.d << "] among " << outputs.size()
                         << " outputs";
  ASSERT_LE(attn_matches, 1) << "ambiguous attention output: " << attn_matches
                             << " tensors match shape [1," << cfg.s << ","
                             << cfg.hq << "," << cfg.d << "]";
  const auto& out_tensor = outputs[attn_idx].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  std::vector<float> golden = load_golden(golden_path, on);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  const bool pass = sdpa_within_tol(
      out_data, golden.data(), on, &max_abs_err, &max_rel_err, cfg.kv_f16);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      on);
  EXPECT_TRUE(pass) << cfg.name
                    << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";
}

// Replay one sequence: thread the op's returned (mutated) KV cache across
// steps, comparing each step's attention output to its accumulated-context
// golden.
void test_sdpa_replay(const SdpaSequence& seq, const std::string& dir) {
  printf(
      "\n--- Test: sdpa replay (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %zu steps) ---\n",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      seq.seq_lens.size());
  if (seq.kv_f16 && !shader_f16_supported_on_test_device()) {
    printf("SKIP: %s requires shader-f16\n", seq.name);
    return;
  }

  const int cn = seq.cmax * seq.hkv * seq.d;
  std::vector<float> kc(cn, 0.0f), vc(cn, 0.0f);
  int input_pos = 0;
  int k_idx = -1,
      v_idx = -1; // pinned at step 0 by content (caches share numel)

  for (size_t t = 0; t < seq.seq_lens.size(); t++) {
    const int s = seq.seq_lens[t];
    const std::string base = dir + "sdpa_" + seq.name + "_step" +
        std::to_string(t) + "_S" + std::to_string(s) + "_pos" +
        std::to_string(input_pos);
    Module module(base + ".pte");
    ASSERT_EQ(load_sdpa_forward(module, seq.kv_f16), Error::Ok)
        << "could not load " << base << ".pte";

    const int qn = s * seq.hq * seq.d;
    const int kvn = s * seq.hkv * seq.d;
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, static_cast<int>(t));
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, static_cast<int>(t));
      v[i] = sdpa_ramp_t(i, 11, 5, static_cast<int>(t));
    }

    auto qt = make_tensor_ptr({1, s, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, s, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, s, seq.hkv, seq.d}, std::vector<float>(v));
    auto kct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(kc));
    auto vct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(vc));

    auto result = module.forward(
        {EValue(qt), EValue(kt), EValue(vt), EValue(kct), EValue(vct)});
    ASSERT_TRUE(result.ok())
        << "forward " << base << ".pte (error " << (int)result.error() << ")";
    if (seq.kv_f16) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      // S==1 decode -> flash-decoding; a multi-token exact-Qwen3-geometry
      // prefill -> the K16 streaming (causal-bound) route by default (no env);
      // any other multi-token fp16-KV shape -> materialized.
      const bool qwen3_geometry = seq.hq == 16 && seq.hkv == 8 && seq.d == 128;
      const bool qwen3_streaming =
          qwen3_geometry && qwen3_q16_supported_on_test_device();
      const uint32_t expected_route = s == 1 ? kTestRouteFlashDecoding
          : qwen3_streaming                  ? kTestRouteK16CausalBound
                                             : kTestRouteMaterializedAttention;
      EXPECT_EQ(
          g_last_route_mask &
              (kTestRouteMaterializedAttention | kTestRouteFlashDecoding |
               kTestRouteK16CausalBound),
          expected_route)
          << seq.name << " step" << t;
      EXPECT_EQ(g_last_route_conflict_count, 0u) << seq.name << " step" << t;
#endif // WGPU_BACKEND_ENABLE_PROFILING
    }
    const auto& outs = result.get();

    // The op returns [k_cache, v_cache, attn_output]: attn has a unique numel;
    // the two caches share numel cn, so identify them by content at step 0.
    int attn_idx = -1;
    std::vector<int> cache_idxs;
    for (size_t i = 0; i < outs.size(); i++) {
      if (!outs[i].isTensor()) {
        continue;
      }
      const int ne = static_cast<int>(outs[i].toTensor().numel());
      if (ne == qn) {
        attn_idx = static_cast<int>(i);
      } else if (ne == cn) {
        cache_idxs.push_back(static_cast<int>(i));
      }
    }
    ASSERT_TRUE(attn_idx >= 0 && cache_idxs.size() == 2)
        << seq.name << " step" << t << ": expected 1 attn + 2 caches";

    if (t == 0) {
      const float* c0 = outs[cache_idxs[0]].toTensor().const_data_ptr<float>();
      const float* c1 = outs[cache_idxs[1]].toTensor().const_data_ptr<float>();
      auto rows_match = [&](const float* c, const std::vector<float>& src) {
        for (int i = 0; i < kvn; i++) {
          if (std::abs(c[i] - src[i]) > 1e-6f) {
            return false;
          }
        }
        return true;
      };
      if (rows_match(c0, k) && rows_match(c1, v)) {
        k_idx = cache_idxs[0];
        v_idx = cache_idxs[1];
      } else if (rows_match(c1, k) && rows_match(c0, v)) {
        k_idx = cache_idxs[1];
        v_idx = cache_idxs[0];
      } else {
        FAIL() << seq.name << " step0 cannot identify k/v cache by content";
      }
      printf("  k/v cache outputs: k_idx=%d v_idx=%d\n", k_idx, v_idx);
    }

    std::vector<float> golden = load_golden(base + ".golden.bin", qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << base << ".golden.bin";
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok =
        sdpa_within_tol(ad, golden.data(), qn, &ma, &mr, seq.kv_f16);
    printf(
        "  step%zu (S=%d pos=%d ctx=%d): max abs %e  rel %e\n",
        t,
        s,
        input_pos,
        input_pos + s,
        ma,
        mr);
    EXPECT_TRUE(step_ok)
        << seq.name << " step" << t
        << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";

    // Thread the device-written caches into the next step (K->K, V->V).
    const float* kd = outs[k_idx].toTensor().const_data_ptr<float>();
    const float* vd = outs[v_idx].toTensor().const_data_ptr<float>();
    kc.assign(kd, kd + cn);
    vc.assign(vd, vd + cn);
    input_pos += s;
  }
}

// Dynamic input_pos decode: ONE .pte (S=1, runtime SymInt input_pos) reused
// across decode steps. Each forward() supplies input_pos as a [1] int64 tensor;
// the backend reads it (update_symints_from_inputs) and recomputes dispatch
// state (propagate_resize) before replaying. The cache is threaded host-side
// (the Module re-copies inputs each call), so correctness hinges on the
// per-step input_pos actually being read + applied. negative=true pins
// input_pos at 0 every step (stale context_len) and asserts the run DIVERGES,
// proving the runtime input_pos + resize hook are load-bearing (no false-pass).
void test_sdpa_dynamic_decode(
    const SdpaSequence& seq,
    const std::string& dir,
    bool negative) {
  constexpr int kSteps = 6; // mirrors DYN_DECODE_STEPS in test_sdpa.py
  printf(
      "\n--- Test: sdpa dynamic decode%s (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %d steps) ---\n",
      negative ? " [NEGATIVE]" : "",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      kSteps);

  const std::string pte = dir + "sdpa_dyn_" + seq.name + ".pte";
  Module module(pte);
  ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load " << pte;

  const int cn = seq.cmax * seq.hkv * seq.d;
  std::vector<float> kc(cn, 0.0f), vc(cn, 0.0f);
  int k_idx = -1,
      v_idx = -1; // pinned at step 0 by content (caches share numel)
  bool any_mismatch = false;

  for (int t = 0; t < kSteps; t++) {
    const int qn = seq.hq * seq.d; // S=1
    const int kvn = seq.hkv * seq.d; // S=1
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, t);
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, t);
      v[i] = sdpa_ramp_t(i, 11, 5, t);
    }
    auto qt = make_tensor_ptr({1, 1, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(v));
    auto kct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(kc));
    auto vct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(vc));
    const int64_t pos = negative ? 0 : t;
    auto ipt = make_tensor_ptr({1}, std::vector<int64_t>{pos});

    auto result = module.forward(
        {EValue(qt),
         EValue(kt),
         EValue(vt),
         EValue(kct),
         EValue(vct),
         EValue(ipt)});
    ASSERT_TRUE(result.ok())
        << "forward step" << t << " (error " << (int)result.error() << ")";
    const auto& outs = result.get();

    int attn_idx = -1;
    std::vector<int> cache_idxs;
    for (size_t i = 0; i < outs.size(); i++) {
      if (!outs[i].isTensor()) {
        continue;
      }
      const int ne = static_cast<int>(outs[i].toTensor().numel());
      if (ne == qn) {
        attn_idx = static_cast<int>(i);
      } else if (ne == cn) {
        cache_idxs.push_back(static_cast<int>(i));
      }
    }
    ASSERT_TRUE(attn_idx >= 0 && cache_idxs.size() == 2)
        << seq.name << " step" << t << ": expected 1 attn + 2 caches";
    if (t == 0) {
      const float* c0 = outs[cache_idxs[0]].toTensor().const_data_ptr<float>();
      const float* c1 = outs[cache_idxs[1]].toTensor().const_data_ptr<float>();
      auto rows_match = [&](const float* c, const std::vector<float>& src) {
        for (int i = 0; i < kvn; i++) {
          if (std::abs(c[i] - src[i]) > 1e-6f) {
            return false;
          }
        }
        return true;
      };
      if (rows_match(c0, k) && rows_match(c1, v)) {
        k_idx = cache_idxs[0];
        v_idx = cache_idxs[1];
      } else if (rows_match(c1, k) && rows_match(c0, v)) {
        k_idx = cache_idxs[1];
        v_idx = cache_idxs[0];
      } else {
        FAIL() << seq.name << " step0 cannot identify k/v cache";
      }
    }

    const std::string gpath = dir + "sdpa_dyn_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << gpath;
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok = sdpa_within_tol(ad, golden.data(), qn, &ma, &mr);
    printf(
        "  step%d (pos=%d ctx=%d): max abs %e  rel %e%s\n",
        t,
        (int)pos,
        t + 1,
        ma,
        mr,
        step_ok ? "" : "  <-- mismatch");
    if (!step_ok) {
      any_mismatch = true;
    }

    const float* kd = outs[k_idx].toTensor().const_data_ptr<float>();
    const float* vd = outs[v_idx].toTensor().const_data_ptr<float>();
    kc.assign(kd, kd + cn);
    vc.assign(vd, vd + cn);
  }

  if (negative) {
    // The negative control must DIVERGE: a stale input_pos=0 every step cannot
    // match the accumulating golden -- otherwise the oracle has no teeth.
    EXPECT_TRUE(any_mismatch)
        << seq.name
        << " negative control matched the golden (oracle has no teeth)";
    if (any_mismatch) {
      printf(
          "PASS: sdpa dynamic decode NEGATIVE (%s): stale input_pos diverges "
          "as expected\n",
          seq.name);
    }
    return;
  }
  EXPECT_FALSE(any_mismatch)
      << seq.name << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";
  if (!any_mismatch) {
    printf("PASS: sdpa dynamic decode (%s)\n", seq.name);
  }
}

// In-graph mutable KV cache: ONE .pte whose k_cache/v_cache are mutable buffers
// (NOT forward inputs); the decode loop feeds only the new token (q/k/v, S=1) +
// runtime input_pos, and the cache accumulates in-graph across forward() calls
// (no host threading). fresh_per_step is the static control: reloading the
// Module each step re-seeds the cache to zeros, so it MUST diverge from the
// accumulating golden at step>=1. Persistent-matches + fresh-diverges = proof
// the pass comes from real accumulation, not a static artifact.
void test_sdpa_incache_decode(
    const SdpaSequence& seq,
    const std::string& dir,
    bool fresh_per_step) {
  constexpr int kSteps = 6; // mirrors DYN_DECODE_STEPS in test_sdpa.py
  printf(
      "\n--- Test: sdpa in-graph-cache decode%s (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %d steps) ---\n",
      fresh_per_step ? " [STATIC CONTROL: fresh Module/step]" : "",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      kSteps);

  const std::string pte = dir + "sdpa_incache_" + seq.name + ".pte";
  std::unique_ptr<Module> persistent;
  if (!fresh_per_step) {
    persistent = std::make_unique<Module>(pte);
    ASSERT_EQ(persistent->load_forward(), Error::Ok)
        << "could not load " << pte;
  }

  bool any_mismatch = false;
  for (int t = 0; t < kSteps; t++) {
    const int qn = seq.hq * seq.d; // S=1
    const int kvn = seq.hkv * seq.d; // S=1
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, t);
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, t);
      v[i] = sdpa_ramp_t(i, 11, 5, t);
    }
    auto qt = make_tensor_ptr({1, 1, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(v));
    auto ipt =
        make_tensor_ptr({1}, std::vector<int64_t>{static_cast<int64_t>(t)});

    // Persistent: reuse the one Module (cache accumulates). Fresh: a new Module
    // each step (cache re-seeded to zeros -> no history).
    std::unique_ptr<Module> fresh;
    Module* mod = persistent.get();
    if (fresh_per_step) {
      fresh = std::make_unique<Module>(pte);
      ASSERT_EQ(fresh->load_forward(), Error::Ok) << "could not load " << pte;
      mod = fresh.get();
    }

    // NOTE: only q/k/v + input_pos -- NO cache args (caches are mutable
    // buffers).
    auto result =
        mod->forward({EValue(qt), EValue(kt), EValue(vt), EValue(ipt)});
    ASSERT_TRUE(result.ok())
        << "forward step" << t << " (error " << (int)result.error() << ")";
    const auto& outs = result.get();
    int attn_idx = -1;
    for (size_t i = 0; i < outs.size(); i++) {
      if (outs[i].isTensor() &&
          static_cast<int>(outs[i].toTensor().numel()) == qn) {
        attn_idx = static_cast<int>(i);
        break;
      }
    }
    ASSERT_GE(attn_idx, 0) << seq.name << " step" << t
                           << ": no attn output (numel " << qn << ")";

    const std::string gpath = dir + "sdpa_incache_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << gpath;
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok = sdpa_within_tol(ad, golden.data(), qn, &ma, &mr);
    printf(
        "  step%d (pos=%d ctx=%d): max abs %e  rel %e%s\n",
        t,
        t,
        t + 1,
        ma,
        mr,
        step_ok ? "" : "  <-- mismatch");
    if (!step_ok) {
      any_mismatch = true;
    }
  }

  if (fresh_per_step) {
    // The control must DIVERGE: a fresh Module per step has no accumulated
    // history, so it cannot match the accumulating golden at step>=1.
    EXPECT_TRUE(any_mismatch)
        << seq.name
        << " static control matched the accumulating golden -- "
           "accumulation was not actually exercised (false-pass risk)";
    if (any_mismatch) {
      printf(
          "PASS: in-graph-cache STATIC CONTROL (%s) diverges as expected -- "
          "persistence is load-bearing; the positive pass is real accumulation\n",
          seq.name);
    }
    return;
  }
  EXPECT_FALSE(any_mismatch)
      << seq.name << " in-graph-cache decode exceeds tolerance";
  if (!any_mismatch) {
    printf(
        "PASS: sdpa in-graph-cache decode (%s) -- cache accumulated in-graph "
        "with NO host threading\n",
        seq.name);
  }
}

void exercise_symint_host_inputs(
    WebGPUGraph& graph,
    int symint_id,
    int input_tensor_id) {
  const auto& input_ids = graph.input_ids();
  std::vector<InputData> inputs(input_ids.size());
  int64_t host_value = 5;
  bool found = false;
  for (size_t i = 0; i < input_ids.size(); i++) {
    if (input_ids[i] == input_tensor_id) {
      inputs[i] = {&host_value, sizeof(host_value), true};
      found = true;
    }
  }
  ASSERT_TRUE(found) << "select_as_symint source is not a graph input";

  const auto update_from_host = [&](int64_t value) {
    host_value = value;
    graph.update_symints_from_inputs(inputs);
    return graph.read_symint(symint_id);
  };
  EXPECT_EQ(update_from_host(5), 5);
  EXPECT_EQ(
      update_from_host(std::numeric_limits<int32_t>::min()),
      std::numeric_limits<int32_t>::min());
  EXPECT_EQ(
      update_from_host(std::numeric_limits<int32_t>::max()),
      std::numeric_limits<int32_t>::max());

  const auto expect_out_of_range = [&](int64_t value) {
    ASSERT_EQ(update_from_host(17), 17);
    host_value = value;
    try {
      graph.update_symints_from_inputs(inputs);
      ADD_FAILURE() << "accepted out-of-range select_as_symint value " << value;
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(
          error.what(),
          "select_as_symint: selected value is outside int32 range");
    }
    EXPECT_EQ(graph.read_symint(symint_id), 17)
        << "rejected value changed the live SymInt";
  };
  expect_out_of_range(
      int64_t{std::numeric_limits<int32_t>::min()} - int64_t{1});
  expect_out_of_range(
      int64_t{std::numeric_limits<int32_t>::max()} + int64_t{1});
  expect_out_of_range(INT64_C(1) << 32);
}

void test_symint_input_narrowing() {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  const std::vector<uint32_t> dims = {1u};
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::INT32,
          &dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/0)
          .Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, 0).Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, 0).Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::SymInt, vk::CreateSymInt(fbb, 0).Union()));
  const std::vector<int32_t> args = {0, 1, 2, 3};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(vk::CreateOperatorCallDirect(
      fbb, 0, "et_vk.select_as_symint.default", &args));
  const std::vector<uint32_t> input_ids = {0};
  const std::vector<uint32_t> output_ids = {0};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraph graph;
  ASSERT_NO_THROW(graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr));
  ASSERT_EQ(graph.symint_sources().size(), 1u);
  const auto& source = graph.symint_sources().front();
  exercise_symint_host_inputs(graph, source.symint_id, source.input_tensor_id);
}

void write_u16_le(std::vector<uint8_t>& data, size_t offset, uint16_t value) {
  data.at(offset) = static_cast<uint8_t>(value);
  data.at(offset + 1) = static_cast<uint8_t>(value >> 8);
}

void write_u32_le(std::vector<uint8_t>& data, size_t offset, uint32_t value) {
  for (size_t i = 0; i < sizeof(value); i++) {
    data.at(offset + i) = static_cast<uint8_t>(value >> (8 * i));
  }
}

void write_u64_le(std::vector<uint8_t>& data, size_t offset, uint64_t value) {
  for (size_t i = 0; i < sizeof(value); i++) {
    data.at(offset + i) = static_cast<uint8_t>(value >> (8 * i));
  }
}

std::vector<uint8_t> make_delegate_header_test_blob() {
  std::vector<uint8_t> blob(44, 0);
  std::memcpy(blob.data() + 4, "VH00", 4);
  write_u16_le(blob, 8, 30);
  write_u32_le(blob, 10, 32);
  write_u32_le(blob, 14, 8);
  write_u32_le(blob, 18, 40);
  write_u64_le(blob, 22, 4);
  return blob;
}

void finish_inline_constant_graph(
    ::flatbuffers::FlatBufferBuilder& fbb,
    bool mark_as_kv_cache,
    const std::vector<uint32_t>& dims,
    uint64_t inline_offset = 0) {
  namespace vk = vkgraph;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  const int tensor_count = mark_as_kv_cache ? 5 : 1;
  for (int i = 0; i < tensor_count; i++) {
    const bool is_cache = mark_as_kv_cache && i >= 3;
    const bool is_constant = !mark_as_kv_cache || is_cache;
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb,
            vk::VkDataType::FLOAT32,
            &dims,
            is_constant ? (is_cache ? i - 3 : 0) : -1,
            is_constant ? -1 : i)
            .Union()));
  }

  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  if (mark_as_kv_cache) {
    const std::vector<int32_t> args = {0, 1, 2, 3, 4};
    chain.push_back(vk::CreateOperatorCallDirect(
        fbb, 0, "sdpa_with_kv_cache.default", &args));
  }
  std::vector<::flatbuffers::Offset<vk::VkBytes>> constants;
  constants.push_back(
      vk::CreateVkBytesDirect(fbb, inline_offset, sizeof(float)));
  if (mark_as_kv_cache) {
    constants.push_back(vk::CreateVkBytesDirect(fbb, 0, sizeof(float)));
  }
  const std::vector<uint32_t> output_ids = {0};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, nullptr, &output_ids, &constants);
  vk::FinishVkGraphBuffer(fbb, root);
}

TEST(WebGPUNative, DelegateHeaderRejectsTruncatedRanges) {
  const auto blob = make_delegate_header_test_blob();
  EXPECT_TRUE(WebGPUDelegateHeader::parse(blob.data(), blob.size()).ok());
  EXPECT_FALSE(WebGPUDelegateHeader::parse(blob.data(), 29).ok());
  EXPECT_FALSE(WebGPUDelegateHeader::parse(blob.data(), blob.size() - 1).ok());
}

TEST(WebGPUNative, InlineConstantExtentIsBounded) {
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_inline_constant_graph(fbb, false, {1u});
  const std::array<uint8_t, sizeof(float)> data = {0, 0, 0, 0};

  WebGPUGraph exact_graph;
  EXPECT_NO_THROW(exact_graph.build(
      fbb.GetBufferPointer(), data.data(), data.size(), nullptr));

  WebGPUGraph short_graph;
  EXPECT_THROW(
      short_graph.build(
          fbb.GetBufferPointer(), data.data(), data.size() - 1, nullptr),
      std::runtime_error);
}

TEST(WebGPUNative, ZeroByteInlineConstantOffsetIsBounded) {
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_inline_constant_graph(fbb, false, {0u}, 1);
  const std::array<uint8_t, 1> data = {0};

  WebGPUGraph graph;
  EXPECT_THROW(
      graph.build(fbb.GetBufferPointer(), data.data(), 0, nullptr),
      std::runtime_error);
}

TEST(WebGPUNative, F16KvInlineConstantExtentIsBounded) {
  const auto* context = get_default_webgpu_context();
  if (context == nullptr || !context->shader_f16_supported) {
    GTEST_SKIP() << "shader-f16 unavailable";
  }
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_inline_constant_graph(fbb, true, {1u});
  const std::array<uint8_t, sizeof(float)> data = {0, 0, 0, 0};

  WebGPUGraph graph;
  WebGPUGraphConfig config;
  config.f16_kv_cache = true;
  try {
    graph.build(
        fbb.GetBufferPointer(), data.data(), data.size() - 1, nullptr, config);
    FAIL() << "undersized inline fp16 KV constant was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "WebGPU f16 KV: inline cache constant exceeds constant data");
  }
}

void expect_tensor_extent_error(
    const std::vector<uint32_t>& dims,
    const char* expected_error) {
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_inline_constant_graph(fbb, false, dims);
  const std::array<uint8_t, sizeof(float)> data = {0, 0, 0, 0};
  WebGPUGraph graph;
  try {
    graph.build(fbb.GetBufferPointer(), data.data(), data.size(), nullptr);
    ADD_FAILURE() << "overflowing tensor extent was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), expected_error);
  }
}

TEST(WebGPUNative, TensorExtentOverflowIsRejected) {
  expect_tensor_extent_error(
      {UINT32_MAX, UINT32_MAX, 2u}, "WebGPU: tensor element count overflows");
  expect_tensor_extent_error(
      {UINT32_MAX, UINT32_MAX}, "WebGPU: tensor byte size overflows");
}

struct DelegateBlobView {
  size_t base_offset;
  WebGPUDelegateHeader header;
};

std::optional<DelegateBlobView> find_delegate_blob(
    const std::vector<uint8_t>& blob) {
  constexpr size_t kHeaderSize = 30;
  constexpr size_t kMagicOffset = 4;
  constexpr char kMagic[] = {'V', 'H', '0', '0'};
  if (blob.size() < kHeaderSize) {
    return std::nullopt;
  }

  for (size_t base_offset = 0; base_offset <= blob.size() - kHeaderSize;
       base_offset++) {
    const uint8_t* base = blob.data() + base_offset;
    if (std::memcmp(base + kMagicOffset, kMagic, sizeof(kMagic)) != 0) {
      continue;
    }
    auto header = WebGPUDelegateHeader::parse(base, blob.size() - base_offset);
    if (!header.ok()) {
      continue;
    }

    const uint64_t available = blob.size() - base_offset;
    const auto range_is_in_blob = [available](uint64_t offset, uint64_t size) {
      return offset <= available && size <= available - offset;
    };
    if (!range_is_in_blob(header->flatbuffer_offset, header->flatbuffer_size) ||
        !range_is_in_blob(header->bytes_offset, header->bytes_size)) {
      continue;
    }
    return DelegateBlobView{base_offset, *header};
  }
  return std::nullopt;
}

TEST(WebGPUNative, StructurallyInvalidVkGraphIsRejectedAtLoad) {
  if (g_symint_blob.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SYMINT_BLOB not set";
  }
  FILE* input = std::fopen(g_symint_blob.c_str(), "rb");
  ASSERT_NE(input, nullptr);
  std::fseek(input, 0, SEEK_END);
  const long file_size = std::ftell(input);
  std::fseek(input, 0, SEEK_SET);
  ASSERT_GT(file_size, 0);
  std::vector<uint8_t> blob(static_cast<size_t>(file_size));
  ASSERT_EQ(std::fread(blob.data(), 1, blob.size(), input), blob.size());
  std::fclose(input);

  const auto delegate = find_delegate_blob(blob);
  ASSERT_TRUE(delegate.has_value());
  ASSERT_GE(delegate->header.flatbuffer_size, sizeof(uint32_t));
  const size_t root_offset =
      delegate->base_offset + delegate->header.flatbuffer_offset;
  std::fill_n(blob.begin() + root_offset, sizeof(uint32_t), UINT8_MAX);

  const std::string malformed_path = "/tmp/webgpu_invalid_vkgraph_" +
      std::to_string(reinterpret_cast<uintptr_t>(blob.data())) + ".pte";
  FILE* output = std::fopen(malformed_path.c_str(), "wb");
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(std::fwrite(blob.data(), 1, blob.size(), output), blob.size());
  std::fclose(output);

  Error load_result = Error::Ok;
  {
    Module module(malformed_path);
    load_result = module.load_forward();
  }
  EXPECT_NE(load_result, Error::Ok);
  EXPECT_EQ(std::remove(malformed_path.c_str()), 0);
}

// S1 SymInt round-trip: confirm a dynamic input_pos stays live.
void test_symint_roundtrip(const std::string& blob_path) {
  printf("\n--- Test: symint round-trip (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  ASSERT_NE(f, nullptr) << blob_path << " not present";
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  ASSERT_EQ(rd, blob.size()) << "short read of " << blob_path;

  const auto delegate = find_delegate_blob(blob);
  ASSERT_TRUE(delegate.has_value())
      << "no complete VH00 delegate blob found in " << blob_path;
  const uint8_t* base = blob.data() + delegate->base_offset;
  WebGPUGraph graph;
  try {
    graph.build(
        base + delegate->header.flatbuffer_offset,
        base + delegate->header.bytes_offset,
        delegate->header.bytes_size,
        nullptr);
  } catch (const std::exception& e) {
    FAIL() << "graph build: " << e.what();
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  ASSERT_GE(sid, 0)
      << "no SymInt value deserialized (input_pos should be a SymInt)";
  ASSERT_NE(graph.symint_buffer(sid), nullptr)
      << "SymInt " << sid << " has no live uniform buffer";
  ASSERT_EQ(graph.read_symint(sid), 0)
      << "SymInt " << sid << " placeholder != 0 (got " << graph.read_symint(sid)
      << ")";
  graph.set_symint(sid, 7);
  ASSERT_EQ(graph.read_symint(sid), 7)
      << "set/read round-trip (got " << graph.read_symint(sid) << ")";

  const auto& srcs = graph.symint_sources();
  ASSERT_FALSE(srcs.empty()) << "no select_as_symint source recorded";
  exercise_symint_host_inputs(
      graph, srcs[0].symint_id, srcs[0].input_tensor_id);

  printf(
      "PASS: symint round-trip (SymInt %d: deserialize, live buffer, "
      "set 0->7, execute-read input_pos->5)\n",
      sid);
}

// Group 1: the resize-hook dirty-gating mechanism (no SDPA dependency).
// A hook keyed to a SymInt must run via propagate_resize() iff that SymInt
// changed since the last propagate_resize, and exactly once per change.
void test_resize_hook(const std::string& blob_path) {
  printf("\n--- Test: resize-hook dirty-gating (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  ASSERT_NE(f, nullptr) << blob_path << " not present";
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  ASSERT_EQ(rd, blob.size()) << "short read of " << blob_path;
  const auto delegate = find_delegate_blob(blob);
  ASSERT_TRUE(delegate.has_value())
      << "no complete VH00 delegate blob found in " << blob_path;
  const uint8_t* base = blob.data() + delegate->base_offset;
  WebGPUGraph graph;
  try {
    graph.build(
        base + delegate->header.flatbuffer_offset,
        base + delegate->header.bytes_offset,
        delegate->header.bytes_size,
        nullptr);
  } catch (const std::exception& e) {
    FAIL() << "graph build: " << e.what();
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  ASSERT_GE(sid, 0) << "no SymInt value deserialized";

  int run_count = 0;
  int last_seen = -1;
  graph.add_resize_hook(sid, [&](WebGPUGraph& g) {
    run_count++;
    last_seen = g.read_symint(sid);
  });

  // 1: change 0->3 then propagate -> hook runs once, sees 3.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  ASSERT_TRUE(run_count == 1 && last_seen == 3)
      << "after set(3)+propagate run_count=" << run_count
      << " last_seen=" << last_seen << " (want 1,3)";
  // 2: propagate again with no change -> hook does NOT run.
  graph.propagate_resize();
  ASSERT_EQ(run_count, 1)
      << "propagate with clean dirty-set ran the hook (run_count=" << run_count
      << ")";
  // 3: set to the SAME value -> not dirty -> hook does NOT run.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  ASSERT_EQ(run_count, 1) << "set(same)+propagate ran the hook (run_count="
                          << run_count << ")";
  // 4: change 3->8 then propagate -> hook runs again, sees 8.
  graph.set_symint(sid, 8);
  graph.propagate_resize();
  ASSERT_TRUE(run_count == 2 && last_seen == 8)
      << "after set(8)+propagate run_count=" << run_count
      << " last_seen=" << last_seen << " (want 2,8)";

  printf(
      "PASS: resize-hook dirty-gating (SymInt %d: runs only on change, "
      "once per change; saw 3 then 8)\n",
      sid);
}

// q4gsw embedding_q4gsw on-GPU configs: small + llama1b (env-gated,
// run-if-present).
struct EmbConfig {
  const char* name;
  const char* model_env;
  const char* indices_env;
  const char* golden_env;
  int num_indices;
  int embed;
};
const EmbConfig kEmbConfigs[] = {
    {"small",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_MODEL",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_INDICES",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_GOLDEN",
     4,
     64},
    {"llama1b",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_MODEL",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_INDICES",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_GOLDEN",
     4,
     2048},
};

// Regression: an edge-dialect-serialized integer slice `start` can arrive as a
// Double (e.g. Florence-2 DaViT serialized start=0 as Double 0.0), which once
// threw "slice: dynamic/unsupported start". The Python op-tests can only emit
// an Int start (the serializer keys on the Python runtime type), so this case
// is unreachable from a .pte export -- it must be built natively. Here we
// hand-author a VkGraph flatbuffer whose slice `start` is a Double value, run
// it on the device, and assert the gather matches in[start + i*step]. Two
// cases: a Double 0.0 (identity) and a Double 2.0 (non-zero offset).
static bool test_slice_double_start_case(double start_d, int out_len) {
  namespace vk = vkgraph;
  // Slice x[1, kInLen] along dim 1 with `start` as a Double; out is
  // [1,out_len].
  constexpr int kInLen = 6;
  printf(
      "\n--- Test: slice Double start (start=%.1f -> out[1,%d]) ---\n",
      start_d,
      out_len);

  // Value ids: 0=in tensor, 1=dim(Int), 2=start(Double), 3=end(Int),
  // 4=step(Int), 5=out tensor. dims are uint vectors; tensors take distinct
  // mem_obj_ids so build() allocates a real Storage buffer for each.
  ::flatbuffers::FlatBufferBuilder fbb;

  std::vector<uint32_t> in_dims = {1u, static_cast<uint32_t>(kInLen)};
  std::vector<uint32_t> out_dims = {1u, static_cast<uint32_t>(out_len)};

  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &in_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/0)
          .Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, /*int_val=*/1).Union()));
  // The value under test: `start` serialized as a Double, not an Int.
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::Double,
      vk::CreateDouble(fbb, /*double_val=*/start_d).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::Int,
      vk::CreateInt(fbb, /*int_val=*/kInLen).Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, /*int_val=*/1).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &out_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/1)
          .Union()));

  std::vector<int32_t> args = {0, 1, 2, 3, 4, 5};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(
      vk::CreateOperatorCallDirect(fbb, 0, "aten.slice_copy.Tensor", &args));

  std::vector<uint32_t> input_ids = {0};
  std::vector<uint32_t> output_ids = {5};
  auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraph graph;
  try {
    graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  } catch (const std::exception& e) {
    printf("FAIL: graph build threw: %s\n", e.what());
    return false;
  }

  std::vector<float> in(kInLen);
  for (int i = 0; i < kInLen; i++) {
    in[i] = static_cast<float>(i) + 0.5f;
  }
  std::vector<InputData> inputs(1);
  inputs[0] = {in.data(), in.size() * sizeof(float), false};
  std::vector<float> out(out_len, -1.0f);
  std::vector<OutputData> outputs(1);
  outputs[0] = {out.data(), out.size() * sizeof(float), true};
  try {
    graph.copy_inputs(inputs);
    const WebGPUExecutionPlan plan = graph.make_execution_plan({});
    graph.execute(plan);
    graph.copy_outputs(outputs, plan);
  } catch (const std::exception& e) {
    printf("FAIL: slice execute threw: %s\n", e.what());
    return false;
  }

  const int start = static_cast<int>(start_d);
  float max_abs_err = 0.0f;
  for (int i = 0; i < out_len; i++) {
    const float expected = in[start + i]; // step == 1
    max_abs_err = std::max(max_abs_err, std::abs(out[i] - expected));
  }
  printf("Max abs error: %e (checked %d elements)\n", max_abs_err, out_len);
  if (max_abs_err != 0.0f) { // pure gather: must be bit-exact
    printf("FAIL: slice Double-start gather mismatch\n");
    return false;
  }
  printf("PASS: slice Double start (start=%.1f)\n", start_d);
  return true;
}

// Negative control: a Double `start` that is fractional, NaN, or outside the
// int64 range must throw (never silently truncate, never invoke UB via the
// int64_t cast).
static bool test_slice_double_start_rejects(double bad_start) {
  namespace vk = vkgraph;
  constexpr int kInLen = 6;
  printf("\n--- Test: slice Double start REJECTS (start=%g) ---\n", bad_start);

  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<uint32_t> in_dims = {1u, static_cast<uint32_t>(kInLen)};
  std::vector<uint32_t> out_dims = {1u, static_cast<uint32_t>(kInLen)};

  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &in_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/0)
          .Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, /*int_val=*/1).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::Double,
      vk::CreateDouble(fbb, /*double_val=*/bad_start).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::Int,
      vk::CreateInt(fbb, /*int_val=*/kInLen).Union()));
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, /*int_val=*/1).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &out_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/1)
          .Union()));

  std::vector<int32_t> args = {0, 1, 2, 3, 4, 5};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(
      vk::CreateOperatorCallDirect(fbb, 0, "aten.slice_copy.Tensor", &args));
  std::vector<uint32_t> input_ids = {0};
  std::vector<uint32_t> output_ids = {5};
  auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraph graph;
  try {
    graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  } catch (const std::exception& e) {
    printf("PASS: rejected as expected: %s\n", e.what());
    return true;
  }
  printf(
      "FAIL: expected a throw for start=%g, graph.build() succeeded\n",
      bad_start);
  return false;
}

static bool test_slice_double_start() {
  // start=0.0 (identity copy) + start=2.0 (non-zero gather offset).
  bool ok = true;
  ok = test_slice_double_start_case(/*start_d=*/0.0, /*out_len=*/6) && ok;
  ok = test_slice_double_start_case(/*start_d=*/2.0, /*out_len=*/4) && ok;
  // Reject: fractional, NaN, and out-of-int64-range Doubles.
  ok = test_slice_double_start_rejects(/*bad_start=*/0.5) && ok;
  ok = test_slice_double_start_rejects(
           /*bad_start=*/std::numeric_limits<double>::quiet_NaN()) &&
      ok;
  ok = test_slice_double_start_rejects(/*bad_start=*/1e300) && ok;
  return ok;
}

// Regression for serialized integer select arguments that alias an earlier
// floating-point scalar in the Vulkan scalar cache. A production graph has
// select calls whose dim and index both reference Double 0.0.
static void finish_select_scalar_graph(
    ::flatbuffers::FlatBufferBuilder& fbb,
    double dim,
    double index,
    uint32_t out_len,
    bool symint_dim = false) {
  namespace vk = vkgraph;
  std::vector<uint32_t> in_dims = {2u, 3u};
  std::vector<uint32_t> out_dims = {out_len};

  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &in_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/0)
          .Union()));
  if (symint_dim) {
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::SymInt,
        vk::CreateSymInt(fbb, /*value=*/0).Union()));
  } else {
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::Double,
        vk::CreateDouble(fbb, /*double_val=*/dim).Union()));
  }
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::Double,
      vk::CreateDouble(fbb, /*double_val=*/index).Union()));
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::VkTensor,
      vk::CreateVkTensorDirect(
          fbb,
          vk::VkDataType::FLOAT32,
          &out_dims,
          /*constant_id=*/-1,
          /*mem_obj_id=*/1)
          .Union()));

  std::vector<int32_t> args = {0, 1, 2, 3};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(
      vk::CreateOperatorCallDirect(fbb, 0, "aten.select_copy.int", &args));
  std::vector<uint32_t> input_ids = {0};
  std::vector<uint32_t> output_ids = {3};
  auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);
}

static bool test_select_double_scalar_case(
    double dim,
    double index,
    const std::vector<float>& expected) {
  printf(
      "\n--- Test: select Double scalars (dim=%g, index=%g) ---\n", dim, index);
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_select_scalar_graph(
      fbb, dim, index, static_cast<uint32_t>(expected.size()));

  WebGPUGraph graph;
  try {
    graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  } catch (const std::exception& e) {
    printf("FAIL: graph build threw: %s\n", e.what());
    return false;
  }

  std::vector<float> in = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
  std::vector<InputData> inputs = {
      {in.data(), in.size() * sizeof(float), false}};
  std::vector<float> out(expected.size(), -1.0f);
  std::vector<OutputData> outputs = {
      {out.data(), out.size() * sizeof(float), true}};
  try {
    graph.copy_inputs(inputs);
    const WebGPUExecutionPlan plan = graph.make_execution_plan({});
    graph.execute(plan);
    graph.copy_outputs(outputs, plan);
  } catch (const std::exception& e) {
    printf("FAIL: select execute threw: %s\n", e.what());
    return false;
  }

  if (out != expected) {
    printf("FAIL: select Double-scalar gather mismatch\n");
    return false;
  }
  printf("PASS: select Double scalars\n");
  return true;
}

static bool test_select_scalar_build_error(
    double dim,
    double index,
    const char* expected_error,
    bool symint_dim = false) {
  ::flatbuffers::FlatBufferBuilder fbb;
  finish_select_scalar_graph(fbb, dim, index, /*out_len=*/3u, symint_dim);

  WebGPUGraph graph;
  try {
    graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  } catch (const std::exception& e) {
    const std::string error = e.what();
    if (error.find(expected_error) != std::string::npos) {
      printf("PASS: rejected with expected error: %s\n", e.what());
      return true;
    }
    printf(
        "FAIL: rejection error %s did not contain %s\n",
        e.what(),
        expected_error);
    return false;
  }
  printf("FAIL: expected graph build to reject select scalars\n");
  return false;
}

static bool test_select_double_scalars() {
  constexpr double kInt64Limit = 0x1p63;
  const double below_int64_min =
      std::nextafter(-kInt64Limit, -std::numeric_limits<double>::infinity());

  bool ok = true;
  // Production artifact representation: both schema-int arguments alias 0.0.
  ok = test_select_double_scalar_case(
           /*dim=*/0.0, /*index=*/0.0, {0.5f, 1.5f, 2.5f}) &&
      ok;
  // Negative Double indices retain normal select index normalization.
  ok = test_select_double_scalar_case(
           /*dim=*/1.0, /*index=*/-1.0, {2.5f, 5.5f}) &&
      ok;

  ok = test_select_scalar_build_error(
           /*dim=*/0.5,
           /*index=*/0.0,
           "select: non-integral dim") &&
      ok;
  ok = test_select_scalar_build_error(
           /*dim=*/0.0,
           /*index=*/0.5,
           "select: non-integral index") &&
      ok;
  ok = test_select_scalar_build_error(
           std::numeric_limits<double>::quiet_NaN(),
           /*index=*/0.0,
           "select: non-integral dim") &&
      ok;
  ok = test_select_scalar_build_error(
           std::numeric_limits<double>::infinity(),
           /*index=*/0.0,
           "select: non-integral dim") &&
      ok;
  ok = test_select_scalar_build_error(
           kInt64Limit, /*index=*/0.0, "select: non-integral dim") &&
      ok;
  ok = test_select_scalar_build_error(
           below_int64_min, /*index=*/0.0, "select: non-integral dim") &&
      ok;
  // -2^63 is representable: conversion succeeds, then the normal dim check
  // rejects it as out of range for this rank-2 input.
  ok = test_select_scalar_build_error(
           -kInt64Limit, /*index=*/0.0, "select: dim out of range") &&
      ok;
  ok = test_select_scalar_build_error(
           /*dim=*/0.0,
           /*index=*/0.0,
           "select: dynamic/unsupported dim",
           /*symint_dim=*/true) &&
      ok;
  return ok;
}

void expect_rope_hf_resize_numel_overflow(uint32_t q_heads, uint32_t k_heads) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;

  const std::vector<uint32_t> q_dims = {1u, 1u, q_heads, 2u};
  const std::vector<uint32_t> k_dims = {1u, 1u, k_heads, 2u};
  const std::vector<uint32_t> freqs_dims = {2u, 2u};
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  const auto add_tensor = [&](const std::vector<uint32_t>& dims, int mem_id) {
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb,
            vk::VkDataType::FLOAT32,
            &dims,
            /*constant_id=*/-1,
            /*mem_obj_id=*/mem_id)
            .Union()));
  };
  add_tensor(q_dims, 0);
  add_tensor(k_dims, 1);
  add_tensor(freqs_dims, 2);
  add_tensor(freqs_dims, 3);
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, 0).Union()));
  add_tensor(q_dims, 4);
  add_tensor(k_dims, 5);
  const std::vector<int32_t> output_items = {5, 6};
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::ValueList,
      vk::CreateValueListDirect(fbb, &output_items).Union()));

  const std::vector<int32_t> args = {0, 1, 2, 3, 4, 7};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(vk::CreateOperatorCallDirect(
      fbb, 0, "et_vk.apply_rotary_emb_hf.default", &args));
  const std::vector<uint32_t> input_ids = {0, 1, 2, 3};
  const std::vector<uint32_t> output_ids = {5, 6};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraph graph;
  ASSERT_NO_THROW(graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr));
  ASSERT_EQ(graph.num_dispatches(), 2u);
  const uint32_t q_x = graph.dispatch_at(0).workgroup_count_x;
  const uint32_t q_y = graph.dispatch_at(0).workgroup_count_y;
  const uint32_t k_x = graph.dispatch_at(1).workgroup_count_x;
  const uint32_t k_y = graph.dispatch_at(1).workgroup_count_y;

  constexpr int64_t kLargeBatch = INT64_C(1) << 30;
  const std::vector<int64_t> q_live = {
      kLargeBatch, 1, static_cast<int64_t>(q_heads), 2};
  const std::vector<int64_t> k_live = {
      kLargeBatch, 1, static_cast<int64_t>(k_heads), 2};
  graph.get_tensor(0).dims = q_live;
  graph.get_tensor(1).dims = k_live;
  ASSERT_NO_THROW(graph.resize_input(0, q_live));
  ASSERT_NO_THROW(graph.resize_input(1, k_live));

  try {
    graph.propagate_resize();
    FAIL() << "accepted q/k element count outside uint32 range";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "apply_rotary_emb_hf(resize): element index exceeds uint32 range");
  }
  EXPECT_EQ(graph.dispatch_at(0).workgroup_count_x, q_x);
  EXPECT_EQ(graph.dispatch_at(0).workgroup_count_y, q_y);
  EXPECT_EQ(graph.dispatch_at(1).workgroup_count_x, k_x);
  EXPECT_EQ(graph.dispatch_at(1).workgroup_count_y, k_y);
}

// apply_rotary_emb on-GPU configs: multi + decode (env-gated, run-if-present).
struct RopeConfig {
  const char* name;
  const char* model_env;
  const char* xq_env;
  const char* xk_env;
  int S;
  int NH;
  int NKV;
  int HD;
};
const RopeConfig kRopeConfigs[] = {
    {"multi",
     "WEBGPU_TEST_ROPE_MODEL",
     "WEBGPU_TEST_ROPE_XQ_GOLDEN",
     "WEBGPU_TEST_ROPE_XK_GOLDEN",
     5,
     8,
     2,
     64},
    {"decode",
     "WEBGPU_TEST_ROPE_DECODE_MODEL",
     "WEBGPU_TEST_ROPE_DECODE_XQ_GOLDEN",
     "WEBGPU_TEST_ROPE_DECODE_XK_GOLDEN",
     1,
     32,
     8,
     64},
};

} // namespace

#ifdef WGPU_BACKEND_ENABLE_PROFILING
TEST(WebGPUNative, QueryPoolOverrunThrows) {
  test_query_pool_overrun_throws();
}

TEST(WebGPUNative, QueryPoolRoundtrip) {
  test_query_pool_roundtrip(*get_default_webgpu_context());
}

TEST(WebGPUNative, QueryPoolDeltaMath) {
  test_query_pool_delta_math();
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

// The override wg_size must not change results: run the rms_norm scalar kernel
// at wg_size 64 and 128 (the handler always clamps to 64, so 128 is only
// reachable via a direct pipeline) and require element-wise agreement. Absolute
// rms_norm correctness is covered by the model-driven golden tests; this locks
// the runtime-configurability guarantee (same WGSL, different size -> same
// output).
TEST(WebGPUNative, RmsNormWorkgroupSizeConfigurable) {
  const WebGPUContext* ctx = get_default_webgpu_context();
  if (ctx == nullptr || ctx->device == nullptr) {
    GTEST_SKIP() << "no WebGPU device";
  }
  constexpr uint32_t num_rows = 3, row_width = 256;
  constexpr float epsilon = 1e-5f;
  std::vector<float> input(static_cast<size_t>(num_rows) * row_width);
  std::vector<float> weight(row_width);
  for (size_t i = 0; i < input.size(); i++) {
    input[i] = std::sin(0.1f * static_cast<float>(i)) * 2.0f + 0.5f;
  }
  for (uint32_t j = 0; j < row_width; j++) {
    weight[j] = 0.5f + 0.001f * static_cast<float>(j);
  }

  const std::vector<float> out64 =
      run_rms_norm_at_wg(*ctx, 64, input, weight, num_rows, row_width, epsilon);
  const std::vector<float> out128 = run_rms_norm_at_wg(
      *ctx, 128, input, weight, num_rows, row_width, epsilon);
  ASSERT_EQ(out64.size(), static_cast<size_t>(num_rows) * row_width);
  ASSERT_EQ(out128.size(), out64.size());

  double sumsq = 0.0;
  for (float v : out64) {
    sumsq += static_cast<double>(v) * static_cast<double>(v);
  }
  float max_abs = 0.0f, max_rel = 0.0f;
  const bool consistent = sdpa_within_tol(
      out64.data(),
      out128.data(),
      static_cast<int>(out64.size()),
      &max_abs,
      &max_rel);
  printf(
      "  rms_norm wg64-vs-wg128: max_abs=%e max_rel=%e sumsq=%f\n",
      max_abs,
      max_rel,
      sumsq);
  EXPECT_GT(sumsq, 0.0) << "wg64 output is all zero (kernel did not run)";
  EXPECT_TRUE(consistent) << "wg64 vs wg128 differ beyond tol (abs " << max_abs
                          << " rel " << max_rel << ")";
}

TEST(WebGPUNative, UpdateCache) {
  if (g_update_cache_model_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_UPDATE_CACHE_MODEL not set";
  }
  test_update_cache(g_update_cache_model_path);
}

// Guard python<->C++ ramp bit-identity: q4gsw_ramp(0) = -0.5 exactly.
TEST(WebGPUNative, Q4gswRampBitIdentity) {
  EXPECT_LT(std::abs(q4gsw_ramp(0) - (-0.5f)), 1e-12f)
      << "q4gsw_ramp bit-identity check";
}

// q4gsw sweep: self-discover q4gsw_<name>.pte; required=FAIL, heavy=gate.
TEST(WebGPUNative, QuantizedLinearSweep) {
  const std::string& dir = g_qlinear_dir;
  const bool heavy_run = std::getenv("WEBGPU_TEST_HEAVY") != nullptr;
  bool ran = false;
  for (const auto& cfg : kQ4gswConfigs) {
    const std::string pte = dir + "q4gsw_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      if (cfg.required && !dir.empty()) {
        ADD_FAILURE() << "required q4gsw config " << cfg.name
                      << " has no .pte in " << dir;
      }
      continue;
    }
    std::fclose(f);
    if (cfg.heavy && !heavy_run) {
      printf(
          "SKIP: heavy q4gsw config %s (set WEBGPU_TEST_HEAVY=1 on a real GPU)\n",
          cfg.name);
      continue;
    }
    const std::string golden = dir + "q4gsw_" + cfg.name + ".golden.bin";
    ran = true;
    test_q4gsw_config(cfg, pte, golden);
  }
  if (!dir.empty() && !ran) {
    ADD_FAILURE()
        << "WEBGPU_TEST_QUANTIZED_LINEAR_DIR set but no q4gsw config ran";
  }
}

TEST(WebGPUNative, EmbeddingQ4gsw) {
  bool any = false;
  for (const auto& c : kEmbConfigs) {
    const char* m = std::getenv(c.model_env);
    const char* ip = std::getenv(c.indices_env);
    const char* g = std::getenv(c.golden_env);
    if (m && ip && g && *m && *ip && *g) {
      any = true;
      test_embedding_q4gsw(m, ip, g, c.num_indices, c.embed, c.name);
    }
  }
  if (!any) {
    GTEST_SKIP() << "no embedding_q4gsw config env set";
  }
}

TEST(WebGPUNative, Rope) {
  bool any = false;
  for (const auto& c : kRopeConfigs) {
    const char* m = std::getenv(c.model_env);
    const char* xq = std::getenv(c.xq_env);
    const char* xk = std::getenv(c.xk_env);
    if (m && xq && xk && *m && *xq && *xk) {
      any = true;
      test_rope(m, xq, xk, c.S, c.NH, c.NKV, c.HD, c.name);
    }
  }
  if (!any) {
    GTEST_SKIP() << "no apply_rotary_emb config env set";
  }
}

TEST(WebGPUNative, RopeHfDynamic) {
  const char* env = std::getenv("WEBGPU_TEST_ROPE_HF_DIR");
  if (env == nullptr || *env == '\0') {
    GTEST_SKIP() << "WEBGPU_TEST_ROPE_HF_DIR not set";
  }
  std::string dir = env;
  if (dir.back() != '/') {
    dir += '/';
  }
  test_rope_hf_dynamic(dir);
}

TEST(WebGPUNative, RopeHfDynamicSequenceReusedGraph) {
  const char* env = std::getenv("WEBGPU_TEST_ROPE_HF_DIR");
  if (env == nullptr || *env == '\0') {
    GTEST_SKIP() << "WEBGPU_TEST_ROPE_HF_DIR not set";
  }
  std::string dir = env;
  if (dir.back() != '/') {
    dir += '/';
  }
  test_rope_hf_dynamic_sequence_reused_graph(dir);
}

TEST(WebGPUNative, RopeHfUsesFull2DGridStride) {
  const WebGPUContext* ctx = get_default_webgpu_context();
  ASSERT_NE(ctx, nullptr);
  const std::vector<float> output = run_rope_hf_2d_probe(*ctx);
  ASSERT_EQ(output.size(), 16u) << "HF RoPE probe output map failed";
  for (size_t i = 0; i < output.size(); i++) {
    const float expected =
        static_cast<float>(i + 1u) * (i < output.size() / 2u ? 1.0f : 2.0f);
    EXPECT_EQ(output[i], expected)
        << "HF RoPE 2D grid or second-half frequency mismatch at element " << i;
  }
}

TEST(WebGPUNative, RopeHfResizeRejectsQOrKNumelOverflow) {
  expect_rope_hf_resize_numel_overflow(/*q_heads=*/2, /*k_heads=*/1);
  expect_rope_hf_resize_numel_overflow(/*q_heads=*/1, /*k_heads=*/2);
}

TEST(WebGPUNative, Prepack) {
  if (g_prepack_model_path.empty() || g_prepack_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK_MODEL/GOLDEN not set";
  }
  test_prepack(g_prepack_model_path, g_prepack_golden_path);
}

TEST(WebGPUNative, SliceDoubleStart) {
  EXPECT_TRUE(test_slice_double_start())
      << "slice Double-start gather/reject checks failed";
}

TEST(WebGPUNative, SelectDoubleScalars) {
  EXPECT_TRUE(test_select_double_scalars())
      << "select Double-scalar compatibility checks failed";
}

TEST(WebGPUNative, Prepack2) {
  if (g_prepack2_model_path.empty() || g_prepack2_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK2_MODEL/GOLDEN not set";
  }
  test_prepack(g_prepack2_model_path, g_prepack2_golden_path, "x + w1 + w2");
}

TEST(WebGPUNative, PrepackTied) {
  if (g_prepack_tied_model_path.empty() || g_prepack_tied_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK_TIED_MODEL/GOLDEN not set";
  }
  test_prepack(
      g_prepack_tied_model_path,
      g_prepack_tied_golden_path,
      "x + w + w (tied weights, shared key)");
}

// SDPA sweep: configs self-discover sdpa_<name>.pte; required=FAIL else skip.
TEST(WebGPUNative, Qwen3SdpaFixtureContract) {
  const auto find_config = [](const char* name) {
    return std::find_if(
        std::begin(kSdpaConfigs),
        std::end(kSdpaConfigs),
        [name](const SdpaConfig& cfg) {
          return std::strcmp(cfg.name, name) == 0;
        });
  };
  const auto prefill = find_config("qwen3_prefill");
  const auto boundary = find_config("qwen3_odd_boundary");
  ASSERT_NE(prefill, std::end(kSdpaConfigs));
  ASSERT_NE(boundary, std::end(kSdpaConfigs));
  EXPECT_EQ(
      std::vector<int>(
          {prefill->hq,
           prefill->hkv,
           prefill->d,
           prefill->s,
           prefill->cmax,
           prefill->input_pos}),
      std::vector<int>({16, 8, 128, 128, 256, 0}));
  EXPECT_EQ(
      std::vector<int>(
          {boundary->hq,
           boundary->hkv,
           boundary->d,
           boundary->s,
           boundary->cmax,
           boundary->input_pos}),
      std::vector<int>({16, 8, 128, 17, 64, 31}));
  EXPECT_TRUE(prefill->kv_f16 && boundary->kv_f16);

  const auto replay = std::find_if(
      std::begin(kSdpaSequences),
      std::end(kSdpaSequences),
      [](const SdpaSequence& seq) {
        return std::strcmp(seq.name, "qwen3_fd") == 0;
      });
  ASSERT_NE(replay, std::end(kSdpaSequences));
  EXPECT_EQ(
      std::vector<int>({replay->hq, replay->hkv, replay->d, replay->cmax}),
      std::vector<int>({16, 8, 128, 64}));
  EXPECT_EQ(replay->seq_lens, std::vector<int>({17, 1}));
  EXPECT_TRUE(replay->kv_f16);
}

TEST(WebGPUNative, Qwen3SdpaRoutes) {
  if (g_sdpa_dir.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SDPA_DIR not set";
  }
  if (!qwen3_q16_supported_on_test_device()) {
    GTEST_SKIP() << "Qwen3 Q16 K16 device limits unavailable";
  }
  // Default route: exact-Qwen3-geometry fp16-KV configs select the Q16 K16
  // streaming (causal-bound) route by geometry (no runtime config needed) --
  // the per-config assertions live in test_sdpa_config / test_sdpa_replay.
  for (const auto& cfg : kSdpaConfigs) {
    if (std::strncmp(cfg.name, "qwen3_", 6) != 0) {
      continue;
    }
    const std::string base = g_sdpa_dir + "sdpa_" + cfg.name;
    test_sdpa_config(cfg, base + ".pte", base + ".golden.bin");
  }
  const auto replay = std::find_if(
      std::begin(kSdpaSequences),
      std::end(kSdpaSequences),
      [](const SdpaSequence& seq) {
        return std::strcmp(seq.name, "qwen3_fd") == 0;
      });
  ASSERT_NE(replay, std::end(kSdpaSequences));
  test_sdpa_replay(*replay, g_sdpa_dir);

  // Run Q32 over both an aligned prefill and the S=17/nonzero-position case so
  // the partial final workgroup's row mask is covered. Unsupported Q32 devices
  // intentionally fall back to the already-qualified Q16 route.
  for (const auto& cfg : kSdpaConfigs) {
    if (std::strncmp(cfg.name, "qwen3_", 6) != 0) {
      continue;
    }
    const std::string base = g_sdpa_dir + "sdpa_" + cfg.name;
    test_sdpa_config(
        cfg,
        base + ".pte",
        base + ".golden.bin",
        /*sdpa_query_tile=*/32);
  }
}

TEST(WebGPUNative, SdpaSweep) {
  const std::string& dir = g_sdpa_dir;
  bool ran = false;
  for (const auto& cfg : kSdpaConfigs) {
    const std::string pte = dir + "sdpa_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      // required config absent (dir set) = FAIL; otherwise skip silently.
      if (cfg.required && !dir.empty()) {
        ADD_FAILURE() << "required sdpa config " << cfg.name
                      << " has no .pte in " << dir;
      }
      continue; // not embedded in this binary
    }
    std::fclose(f);
    const std::string golden = dir + "sdpa_" + cfg.name + ".golden.bin";
    ran = true;
    test_sdpa_config(cfg, pte, golden);
  }
  if (!dir.empty() && !ran) {
    ADD_FAILURE() << "WEBGPU_TEST_SDPA_DIR set but no sdpa config found a .pte";
  }
}

// Guard python<->C++ ramp bit-identity (recorded: _ramp_t(0,17,8,2)=0.1875).
TEST(WebGPUNative, SdpaRampTBitIdentity) {
  EXPECT_LT(std::abs(sdpa_ramp_t(0, 17, 8, 2) - 0.1875f), 1e-12f)
      << "sdpa_ramp_t bit-identity check";
}

// Guard the adversarial denom path: sdpa_ramp(0,17,8,0.5)= -16.0 exactly.
TEST(WebGPUNative, SdpaRampDenomBitIdentity) {
  EXPECT_LT(std::abs(sdpa_ramp(0, 17, 8, 0.5f) - (-16.0f)), 1e-12f)
      << "sdpa_ramp denom bit-identity check";
}

// Replay sweep: run every sequence whose step0 .pte is present.
TEST(WebGPUNative, SdpaReplaySweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string step0 = dir + "sdpa_" + seq.name + "_step0_S" +
        std::to_string(seq.seq_lens[0]) + "_pos0.pte";
    FILE* f = std::fopen(step0.c_str(), "rb");
    if (!f) {
      continue; // sequence not embedded in this binary
    }
    std::fclose(f);
    test_sdpa_replay(seq, dir);
  }
}

// Dynamic decode sweep: positive + negative control per embedded param set.
TEST(WebGPUNative, SdpaDynamicDecodeSweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_dyn_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    test_sdpa_dynamic_decode(seq, dir, /*negative=*/false);
    test_sdpa_dynamic_decode(seq, dir, /*negative=*/true);
  }
}

// In-graph-cache decode sweep: persistent + fresh (static control) per set.
TEST(WebGPUNative, SdpaIncacheDecodeSweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_incache_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/false);
    test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/true);
  }
}

// If an SDPA dir was given, the exports must have produced .ptes for every
// family; a self-skip there means a silent export failure, not a pass.
TEST(WebGPUNative, SdpaAllFamiliesRanWhenDirSet) {
  const std::string& dir = g_sdpa_dir;
  if (dir.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SDPA_DIR not set";
  }
  auto has_glob = [&](const std::string& prefix, const std::string& suffix) {
    for (const auto& seq : kSdpaSequences) {
      const std::string p = dir + prefix + seq.name + suffix;
      FILE* f = std::fopen(p.c_str(), "rb");
      if (f) {
        std::fclose(f);
        return true;
      }
    }
    return false;
  };
  bool sdpa_ran = false;
  for (const auto& cfg : kSdpaConfigs) {
    const std::string pte = dir + "sdpa_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (f) {
      std::fclose(f);
      sdpa_ran = true;
      break;
    }
  }
  const bool replay_ran = [&] {
    for (const auto& seq : kSdpaSequences) {
      const std::string step0 = dir + "sdpa_" + seq.name + "_step0_S" +
          std::to_string(seq.seq_lens[0]) + "_pos0.pte";
      FILE* f = std::fopen(step0.c_str(), "rb");
      if (f) {
        std::fclose(f);
        return true;
      }
    }
    return false;
  }();
  const bool dyn_ran = has_glob("sdpa_dyn_", ".pte");
  const bool incache_ran = has_glob("sdpa_incache_", ".pte");
  EXPECT_TRUE(sdpa_ran && replay_ran && dyn_ran && incache_ran)
      << "WEBGPU_TEST_SDPA_DIR set but an SDPA family found no .pte";
}

TEST(WebGPUNative, SymintRoundtrip) {
  if (g_symint_blob.empty()) {
    test_symint_input_narrowing();
    return;
  }
  test_symint_roundtrip(g_symint_blob);
}

TEST(WebGPUNative, ResizeHook) {
  if (g_symint_blob.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SYMINT_BLOB not set";
  }
  test_resize_hook(g_symint_blob);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  if (const char* env = std::getenv("WEBGPU_TEST_UPDATE_CACHE_MODEL")) {
    g_update_cache_model_path = env;
  }

  // Quantized-linear sweep dir (mirrors WEBGPU_TEST_SDPA_DIR).
  if (const char* env = std::getenv("WEBGPU_TEST_QUANTIZED_LINEAR_DIR")) {
    g_qlinear_dir = env;
    if (!g_qlinear_dir.empty() && g_qlinear_dir.back() != '/') {
      g_qlinear_dir += '/';
    }
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_MODEL")) {
    g_prepack_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_GOLDEN")) {
    g_prepack_golden_path = env;
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK2_MODEL")) {
    g_prepack2_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK2_GOLDEN")) {
    g_prepack2_golden_path = env;
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_TIED_MODEL")) {
    g_prepack_tied_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_TIED_GOLDEN")) {
    g_prepack_tied_golden_path = env;
  }

  // SDPA sweep: configs self-discover their sdpa_<name>.pte/.golden.bin under
  // this directory (default "" = the embedded-file root / cwd). Set
  // WEBGPU_TEST_SDPA_DIR to point at the exported .pte directory (e.g. /tmp/).
  if (const char* env = std::getenv("WEBGPU_TEST_SDPA_DIR")) {
    g_sdpa_dir = env;
    if (!g_sdpa_dir.empty() && g_sdpa_dir.back() != '/') {
      g_sdpa_dir += '/';
    }
  }

  if (const char* env = std::getenv("WEBGPU_TEST_SYMINT_BLOB")) {
    g_symint_blob = env;
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    printf("SKIP: %s\n", e.what());
    return 0;
  }

  set_default_webgpu_context(&ctx);
  printf("WebGPU device acquired (native)\n");

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
