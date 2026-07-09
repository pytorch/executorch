/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// White-box unit tests for WebGPUGraph scratch buffers:
// create_scratch_buffer and the acquire_scratch/release_scratch reuse pool.

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <webgpu/webgpu.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

using namespace executorch::backends::webgpu;

namespace {

// WebGPU context; set from create_webgpu_context() in main() before
// RUN_ALL_TESTS().
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
WGPUInstance g_instance = nullptr;
WGPUDevice g_device = nullptr;
WGPUQueue g_queue = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

struct MapCb {
  std::atomic<WGPUMapAsyncStatus> status{WGPUMapAsyncStatus_Error};
};

void map_cb(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  auto* d = static_cast<MapCb*>(userdata1);
  d->status.store(status, std::memory_order_release);
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
  ci.mode = WGPUCallbackMode_WaitAnyOnly;
  ci.callback = map_cb;
  ci.userdata1 = &cb;
  webgpu_wait(
      instance, wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, nbytes, ci));

  std::vector<float> out(nbytes / sizeof(float));
  if (cb.status.load(std::memory_order_acquire) == WGPUMapAsyncStatus_Success) {
    const void* m = wgpuBufferGetConstMappedRange(staging, 0, nbytes);
    if (m != nullptr) {
      std::memcpy(out.data(), m, nbytes);
    }
    wgpuBufferUnmap(staging);
  }
  wgpuBufferRelease(staging);
  return out;
}

} // namespace

// Tier 1: allocation, zero-size guard, distinct non-null handles.
TEST(ScratchBuffer, Tier1Alloc) {
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer a = g.create_scratch_buffer(64 * sizeof(float));
  WGPUBuffer z = g.create_scratch_buffer(0); // guarded to 4 bytes
  WGPUBuffer b = g.create_scratch_buffer(64 * sizeof(float));
  EXPECT_TRUE(a && z && b && a != b && a != z && b != z);
  // graph dtor releases all three here
}

// Tier 2: host->scratch write, scratch->staging copy, read-back round-trip.
TEST(ScratchBuffer, Tier2Roundtrip) {
  for (int n : {1, 7, 1024}) {
    WebGPUGraph g;
    g.set_device(g_device);
    WGPUBuffer s = g.create_scratch_buffer(n * sizeof(float));
    std::vector<float> in(n);
    for (int i = 0; i < n; i++) {
      in[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }
    wgpuQueueWriteBuffer(g_queue, s, 0, in.data(), n * sizeof(float));
    std::vector<float> back =
        readback(g_instance, g_device, g_queue, s, n * sizeof(float));
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
      max_err = std::max(max_err, std::abs(back[i] - in[i]));
    }
    // pure copy: must be bit-exact
    EXPECT_EQ(max_err, 0.0f) << "n=" << n << " max abs error " << max_err;
  }
}

// Tier 3a: bind scratch as a Storage buffer in a compute pass (its real use).
TEST(ScratchBuffer, Tier3Compute) {
  const int n = 256;
  WebGPUGraph g;
  g.set_device(g_device);
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
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(g_device, &smd);

  WGPUBindGroupLayoutEntry ble = {};
  ble.binding = 0;
  ble.visibility = WGPUShaderStage_Compute;
  ble.buffer.type = WGPUBufferBindingType_Storage;
  WGPUBindGroupLayoutDescriptor bld = {};
  bld.entryCount = 1;
  bld.entries = &ble;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(g_device, &bld);

  WGPUPipelineLayoutDescriptor pld = {};
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g_device, &pld);

  WGPUComputePipelineDescriptor cpd = {};
  cpd.layout = pl;
  cpd.compute.module = shader;
  cpd.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g_device, &cpd);

  WGPUBindGroupEntry bge = {};
  bge.binding = 0;
  bge.buffer = s;
  bge.size = n * sizeof(float);
  WGPUBindGroupDescriptor bgd = {};
  bgd.layout = bgl;
  bgd.entryCount = 1;
  bgd.entries = &bge;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_device, &bgd);

  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, &ed);
  WGPUComputePassDescriptor pd = {};
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, (n + 63) / 64, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
  wgpuQueueSubmit(g_queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  std::vector<float> back =
      readback(g_instance, g_device, g_queue, s, n * sizeof(float));
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    const float expected = static_cast<float>(i) * 2.0f + 1.0f;
    max_err = std::max(max_err, std::abs(back[i] - expected));
  }

  wgpuBindGroupRelease(bg);
  wgpuComputePipelineRelease(pipe);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuShaderModuleRelease(shader);

  EXPECT_EQ(max_err, 0.0f) << "max abs error " << max_err << " (" << n
                           << " elements)";
}

// Tier 3b: many scratch buffers across repeated graphs; dtor must release all.
TEST(ScratchBuffer, Tier3Lifecycle) {
  for (int iter = 0; iter < 50; iter++) {
    WebGPUGraph g;
    g.set_device(g_device);
    for (int k = 0; k < 256; k++) {
      WGPUBuffer b =
          g.create_scratch_buffer(static_cast<size_t>(k % 17) * sizeof(float));
      EXPECT_NE(b, nullptr);
    }
  } // each graph's dtor releases its 256 buffers here
}

// Tier 4: reuse-pool semantics (acquire_scratch / release_scratch /
// ScopedScratch). The pool recycles single-op-lifetime scratch across ops so N
// layers reuse a small constant of buffers instead of N x.

// A released slot is handed back on the next same-size acquire (the reuse win).
TEST(ScratchPool, ReuseAfterRelease) {
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer a = g.acquire_scratch(64 * sizeof(float));
  g.release_scratch(a);
  WGPUBuffer b = g.acquire_scratch(64 * sizeof(float));
  EXPECT_EQ(a, b) << "released slot should be reused for a same-size request";
}

// A still-in_use slot is never handed to a co-live requester (RAW-safety).
TEST(ScratchPool, NoReuseWhileInUse) {
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer a = g.acquire_scratch(64 * sizeof(float));
  WGPUBuffer b = g.acquire_scratch(64 * sizeof(float)); // a not released
  EXPECT_TRUE(a && b && a != b) << "co-live acquires must be distinct buffers";
}

// Best-fit 2x cap: a large free slot must not back a much smaller request, but
// a request it does fit (size in [n, 2n]) reuses it.
TEST(ScratchPool, BestFitSizeCap) {
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer big = g.acquire_scratch(1024 * sizeof(float));
  g.release_scratch(big);
  // 1024*4 bytes is outside [4, 8], so the big slot is ineligible for 4 bytes.
  WGPUBuffer tiny = g.acquire_scratch(4);
  EXPECT_NE(big, tiny)
      << "oversized slot must not back a tiny request (2x cap)";
  g.release_scratch(tiny);
  WGPUBuffer same = g.acquire_scratch(1024 * sizeof(float));
  EXPECT_EQ(big, same) << "an in-range request should reuse the big slot";
}

// WEBGPU_NO_SCRATCH_POOL bypasses the pool -> a fresh buffer every acquire.
TEST(ScratchPool, BypassEnvNoReuse) {
  setenv("WEBGPU_NO_SCRATCH_POOL", "1", 1);
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer a = g.acquire_scratch(64 * sizeof(float));
  g.release_scratch(a); // no-op on the bypass path
  WGPUBuffer b = g.acquire_scratch(64 * sizeof(float));
  unsetenv("WEBGPU_NO_SCRATCH_POOL");
  EXPECT_TRUE(a && b && a != b) << "bypass must allocate a dedicated buffer";
}

// ScopedScratch releases its slot at scope exit, so the next acquire reuses it.
TEST(ScratchPool, ScopedScratchReleasesOnScopeExit) {
  WebGPUGraph g;
  g.set_device(g_device);
  WGPUBuffer first = nullptr;
  {
    WebGPUGraph::ScopedScratch s(&g, g.acquire_scratch(64 * sizeof(float)));
    first = s; // operator WGPUBuffer
    EXPECT_NE(first, nullptr);
  } // s releases the slot here
  WGPUBuffer second = g.acquire_scratch(64 * sizeof(float));
  EXPECT_EQ(first, second) << "slot freed by ScopedScratch should be reused";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    std::printf("SKIP: %s\n", e.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);
  g_instance = ctx.instance;
  g_device = ctx.device;
  g_queue = ctx.queue;

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
