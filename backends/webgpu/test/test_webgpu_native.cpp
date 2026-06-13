/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

static bool test_single_add(const std::string& model_path) {
  printf("\n--- Test: single add (1024x1024) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int dim = 1024;
  constexpr int size = dim * dim;

  std::vector<float> a_data(size);
  std::vector<float> b_data(size);
  for (int i = 0; i < size; i++) {
    a_data[i] = static_cast<float>(i) * 1.0f;
    b_data[i] = static_cast<float>(i) * 2.0f;
  }

  auto a = make_tensor_ptr({dim, dim}, std::vector<float>(a_data));
  auto b = make_tensor_ptr({dim, dim}, std::vector<float>(b_data));

  auto result = module.forward({EValue(a), EValue(b)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }

  const auto& out_tensor = outputs[0].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_error = 0.0f;
  int check_count = std::min(size, 1024);
  for (int i = 0; i < check_count; i++) {
    float expected = a_data[i] + b_data[i];
    float error = std::abs(out_data[i] - expected);
    max_error = std::max(max_error, error);
  }

  printf("Max error: %e (checked %d elements)\n", max_error, check_count);
  if (max_error > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: single add test\n");
  return true;
}

static bool test_chained_add(const std::string& model_path) {
  printf("\n--- Test: chained add (1024x1024, 5 ops) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int dim = 1024;
  constexpr int size = dim * dim;

  std::vector<float> x_data(size);
  std::vector<float> y_data(size);
  for (int i = 0; i < size; i++) {
    x_data[i] = static_cast<float>(i % 100) * 0.01f;
    y_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  auto x = make_tensor_ptr({dim, dim}, std::vector<float>(x_data));
  auto y = make_tensor_ptr({dim, dim}, std::vector<float>(y_data));

  auto result = module.forward({EValue(x), EValue(y)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }

  // z=x+y; z=z+x=2x+y; z=z+y=2x+2y; z=z+x=3x+2y; z=z+y=3x+3y
  const auto& out_tensor = outputs[0].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_error = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = 3.0f * x_data[i] + 3.0f * y_data[i];
    float error = std::abs(out_data[i] - expected);
    max_error = std::max(max_error, error);
  }

  printf("Max error: %e (checked %d elements)\n", max_error, size);
  if (max_error > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: chained add test\n");
  return true;
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
// Capacity-overrun must throw; runs without a device or TimestampQuery.
static bool test_query_pool_overrun_throws() {
  printf("\n--- Test: WebGPUQueryPool capacity-overrun guard ---\n");
  WebGPUQueryPool qp;
  try {
    qp.reset(1);
  } catch (const std::exception&) {
    printf("PASS: reset beyond capacity throws\n");
    return true;
  }
  printf("FAIL: reset beyond capacity did not throw\n");
  return false;
}

// WebGPUQueryPool roundtrip: time a probe pass; assert non-zero GPU duration.
static bool test_query_pool_roundtrip(const WebGPUContext& ctx) {
  printf("\n--- Test: WebGPUQueryPool roundtrip ---\n");
  if (!ctx.timestamp_supported) {
    printf("SKIP: adapter lacks TimestampQuery feature\n");
    return true;
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

  if (qp.results().size() != 1) {
    printf("FAIL: expected 1 duration, got %zu\n", qp.results().size());
    return false;
  }
  const uint64_t dur = qp.results()[0].execution_duration_ns;
  printf("  probe duration: %llu ns\n", (unsigned long long)dur);
  if (dur == 0) {
    printf("FAIL: probe duration is zero (expected monotonic non-zero)\n");
    return false;
  }
  printf("PASS: WebGPUQueryPool roundtrip -- non-zero GPU kernel duration\n");
  return true;
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

int main(int argc, char** argv) {
  std::string model_path = "webgpu_add_test.pte";
  if (argc > 1) {
    model_path = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_TEST_MODEL")) {
    model_path = env;
  }

  std::string chained_model_path;
  if (const char* env = std::getenv("WEBGPU_TEST_CHAINED_MODEL")) {
    chained_model_path = env;
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

  bool ok = true;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  ok = test_query_pool_overrun_throws() && ok;
  ok = test_query_pool_roundtrip(ctx) && ok;
#endif // WGPU_BACKEND_ENABLE_PROFILING
  ok = test_single_add(model_path) && ok;

  if (!chained_model_path.empty()) {
    ok = test_chained_add(chained_model_path) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll tests passed\n");
  return 0;
}
