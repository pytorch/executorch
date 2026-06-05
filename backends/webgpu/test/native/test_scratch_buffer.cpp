/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// White-box unit tests for WebGPUGraph::create_scratch_buffer.

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace executorch::backends::webgpu;

namespace {

struct MapCb {
  bool done = false;
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

void map_cb(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  auto* d = static_cast<MapCb*>(userdata1);
  d->status = status;
  d->done = true;
}

// Copy `src` (must carry CopySrc) into a staging buffer and read it back.
std::vector<float> readback(
    WGPUInstance instance,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUBuffer src,
    size_t nbytes) {
  WGPUBufferDescriptor sd = {};
  sd.size = nbytes;
  sd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &sd);

  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &ed);
  wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, staging, 0, nbytes);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  MapCb cb;
  WGPUBufferMapCallbackInfo ci = {};
  ci.mode = WGPUCallbackMode_AllowSpontaneous;
  ci.callback = map_cb;
  ci.userdata1 = &cb;
  wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, nbytes, ci);
  while (!cb.done) {
    webgpu_poll(instance, device);
  }

  std::vector<float> out(nbytes / sizeof(float));
  if (cb.status == WGPUMapAsyncStatus_Success) {
    const void* m = wgpuBufferGetConstMappedRange(staging, 0, nbytes);
    std::memcpy(out.data(), m, nbytes);
    wgpuBufferUnmap(staging);
  }
  wgpuBufferRelease(staging);
  return out;
}

// Tier 1: allocation, zero-size guard, distinct non-null handles.
bool tier1_alloc(WGPUDevice device) {
  printf("\n--- scratch[tier1: allocation] ---\n");
  WebGPUGraph g;
  g.set_device(device);
  WGPUBuffer a = g.create_scratch_buffer(64 * sizeof(float));
  WGPUBuffer z = g.create_scratch_buffer(0); // guarded to 4 bytes
  WGPUBuffer b = g.create_scratch_buffer(64 * sizeof(float));
  const bool ok = a && z && b && a != b && a != z && b != z;
  printf(ok ? "PASS: allocation\n" : "FAIL: allocation\n");
  return ok; // graph dtor releases all three here
}

// Tier 2: host->scratch write, scratch->staging copy, read-back round-trip.
bool tier2_roundtrip(
    WGPUInstance instance,
    WGPUDevice device,
    WGPUQueue queue) {
  printf("\n--- scratch[tier2: copy round-trip] ---\n");
  bool ok = true;
  for (int n : {1, 7, 1024}) {
    WebGPUGraph g;
    g.set_device(device);
    WGPUBuffer s = g.create_scratch_buffer(n * sizeof(float));
    std::vector<float> in(n);
    for (int i = 0; i < n; i++) {
      in[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }
    wgpuQueueWriteBuffer(queue, s, 0, in.data(), n * sizeof(float));
    std::vector<float> back =
        readback(instance, device, queue, s, n * sizeof(float));
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
      max_err = std::max(max_err, std::abs(back[i] - in[i]));
    }
    printf("  n=%d max abs error %e\n", n, max_err);
    if (max_err != 0.0f) { // pure copy: must be bit-exact
      ok = false;
    }
  }
  printf(ok ? "PASS: copy round-trip\n" : "FAIL: copy round-trip\n");
  return ok;
}

// Tier 3a: bind scratch as a Storage buffer in a compute pass (its real use).
bool tier3_compute(WGPUInstance instance, WGPUDevice device, WGPUQueue queue) {
  printf("\n--- scratch[tier3: compute Storage round-trip] ---\n");
  const int n = 256;
  WebGPUGraph g;
  g.set_device(device);
  WGPUBuffer s = g.create_scratch_buffer(n * sizeof(float));

  const char* kWgsl =
      "@group(0) @binding(0) var<storage, read_write> buf: array<f32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
      "  let i = gid.x;\n"
      "  if (i < arrayLength(&buf)) { buf[i] = f32(i) * 2.0 + 1.0; }\n"
      "}\n";

  WGPUShaderSourceWGSL wgsl = {};
  wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl.code = {kWgsl, WGPU_STRLEN};
  WGPUShaderModuleDescriptor smd = {};
  smd.nextInChain = &wgsl.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &smd);

  WGPUBindGroupLayoutEntry ble = {};
  ble.binding = 0;
  ble.visibility = WGPUShaderStage_Compute;
  ble.buffer.type = WGPUBufferBindingType_Storage;
  WGPUBindGroupLayoutDescriptor bld = {};
  bld.entryCount = 1;
  bld.entries = &ble;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bld);

  WGPUPipelineLayoutDescriptor pld = {};
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pld);

  WGPUComputePipelineDescriptor cpd = {};
  cpd.layout = pl;
  cpd.compute.module = shader;
  cpd.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(device, &cpd);

  WGPUBindGroupEntry bge = {};
  bge.binding = 0;
  bge.buffer = s;
  bge.size = n * sizeof(float);
  WGPUBindGroupDescriptor bgd = {};
  bgd.layout = bgl;
  bgd.entryCount = 1;
  bgd.entries = &bge;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, &ed);
  WGPUComputePassDescriptor pd = {};
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, (n + 63) / 64, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  std::vector<float> back =
      readback(instance, device, queue, s, n * sizeof(float));
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    const float expected = static_cast<float>(i) * 2.0f + 1.0f;
    max_err = std::max(max_err, std::abs(back[i] - expected));
  }
  printf("  max abs error %e (%d elements)\n", max_err, n);

  wgpuBindGroupRelease(bg);
  wgpuComputePipelineRelease(pipe);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuShaderModuleRelease(shader);

  const bool ok = max_err == 0.0f;
  printf(
      ok ? "PASS: compute Storage round-trip\n" : "FAIL: compute round-trip\n");
  return ok;
}

// Tier 3b: many scratch buffers across repeated graphs; dtor must release all.
bool tier3_lifecycle(WGPUDevice device) {
  printf("\n--- scratch[tier3: lifecycle/stress] ---\n");
  bool ok = true;
  for (int iter = 0; iter < 50; iter++) {
    WebGPUGraph g;
    g.set_device(device);
    for (int k = 0; k < 256; k++) {
      WGPUBuffer b =
          g.create_scratch_buffer(static_cast<size_t>(k % 17) * sizeof(float));
      ok = ok && b != nullptr;
    }
  } // each graph's dtor releases its 256 buffers here
  printf(
      ok ? "PASS: lifecycle/stress (50 graphs x 256 buffers)\n"
         : "FAIL: lifecycle/stress (null buffer)\n");
  return ok;
}

} // namespace

int main() {
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
  ok = tier1_alloc(ctx.device) && ok;
  ok = tier2_roundtrip(ctx.instance, ctx.device, ctx.queue) && ok;
  ok = tier3_compute(ctx.instance, ctx.device, ctx.queue) && ok;
  ok = tier3_lifecycle(ctx.device) && ok;

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll scratch_buffer tests passed\n");
  return 0;
}
