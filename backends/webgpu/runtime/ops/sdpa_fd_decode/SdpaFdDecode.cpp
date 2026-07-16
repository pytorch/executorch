/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Split-KV FlashDecoding decode dispatch (split + reduce passes).

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/SdpaFdDecode.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_reduce_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_split_half_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_split_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// MUST match the .wgsl: MAX_SPLITS and WG_SIZE*MAX_D_PER_LANE.
constexpr uint32_t kSdpaFdSplitTile = 64; // KV positions per split
constexpr uint32_t kSdpaFdMaxSplits = 128; // == MAX_SPLITS in both .wgsl files
// Public head-dim limit (kSdpaFdMaxHeadDim) must equal the kernel's lane-owns-D
// reach; tie them so a WG_SIZE change can't silently desync the Sdpa.cpp gate.
static_assert(
    kSdpaFdMaxHeadDim == kSdpaFdSplitWorkgroupSizeX * 2u,
    "kSdpaFdMaxHeadDim must match WG_SIZE * MAX_D_PER_LANE");

struct FdSplitParams {
  uint32_t _pad0; // 16B-alignment pad (head index derived from workgroup_id)
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t g;
  uint32_t num_splits;
  uint32_t split_len;
  float scale;
};
static_assert(sizeof(FdSplitParams) == 32, "FdSplitParams must be 32B");

struct FdReduceParams {
  uint32_t D;
  uint32_t num_splits;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(FdReduceParams) == 16, "FdReduceParams must be 16B");

struct BufferBinding {
  WGPUBuffer buffer;
  uint64_t size;
};

// Mirrors Sdpa.cpp build_dispatch; n_rw leading bindings are read_write.
void build_dispatch(
    WebGPUGraph& graph,
    const char* wgsl_source,
    const BufferBinding* storage_bindings,
    uint32_t n_storage,
    uint32_t n_rw,
    WGPUBuffer uniform_buffer,
    uint64_t uniform_size,
    uint32_t workgroup_count_x,
    const char* kernel_name) {
  WGPUDevice device = graph.device();

  // Storage bindings then the uniform; n_rw leading bindings are read_write.
  // Under auto layout the access class is derived from the shader, so `type`
  // here is descriptive only.
  std::vector<utils::BindingSpec> bindings;
  bindings.reserve(n_storage + 1);
  const uint32_t uniform_binding = n_storage;
  for (uint32_t i = 0; i < n_storage; i++) {
    bindings.push_back(
        {i,
         (i < n_rw) ? WGPUBufferBindingType_Storage
                    : WGPUBufferBindingType_ReadOnlyStorage,
         storage_bindings[i].buffer,
         storage_bindings[i].size});
  }
  bindings.push_back(
      {uniform_binding,
       WGPUBufferBindingType_Uniform,
       uniform_buffer,
       uniform_size});

  utils::ComputePipelineBundle bundle =
      utils::make_compute_pipeline(device, wgsl_source, bindings);

  graph.add_dispatch(
      {bundle.pipeline, bundle.bind_group, workgroup_count_x, kernel_name});

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

void sdpa_fd_decode_dispatch(
    WebGPUGraph& graph,
    const WebGPUTensor& q,
    const WebGPUTensor& k_cache,
    const WebGPUTensor& v_cache,
    const WebGPUTensor& out,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t context_len,
    int64_t g,
    float scale) {
  // Defensive contract guard: the Sdpa.cpp gate only routes D <= this here, but
  // keep the check (lane-owns-D reach) so a future caller can't silently
  // overrun.
  if (D > kSdpaFdMaxHeadDim) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: head dim must be <= " +
        std::to_string(kSdpaFdMaxHeadDim));
  }
  if (D % 4 != 0) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: head dim must be a multiple of 4");
  }
  // context_len 0 -> split_len 0 -> empty KV loop -> silent zero output; the
  // Sdpa.cpp gate guarantees ctx >= 1, but fail loud if called directly.
  if (context_len <= 0) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: context_len must be positive");
  }

  // Split factor: one split per kSdpaFdSplitTile KV rows, capped.
  uint32_t num_splits = static_cast<uint32_t>(
      (context_len + kSdpaFdSplitTile - 1) / kSdpaFdSplitTile);
  if (num_splits > kSdpaFdMaxSplits) {
    num_splits = kSdpaFdMaxSplits;
  }
  const uint32_t split_len =
      static_cast<uint32_t>((context_len + num_splits - 1) / num_splits);

  // Scratch: per-(head,split) partials at kSdpaFdMaxSplits stride.
  const uint64_t po_floats = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(kSdpaFdMaxSplits) * static_cast<uint64_t>(D);
  const uint64_t pml_floats = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(kSdpaFdMaxSplits) * 2ull;
  WGPUBuffer part_o = graph.acquire_scratch(po_floats * sizeof(float));
  WebGPUGraph::ScopedScratch part_o_guard(&graph, part_o);
  WGPUBuffer part_ml = graph.acquire_scratch(pml_floats * sizeof(float));
  WebGPUGraph::ScopedScratch part_ml_guard(&graph, part_ml);

  // Pass 1: split (Hq*num_splits WGs) -> writes part_o, part_ml.
  FdSplitParams sp = {};
  sp.Hkv = static_cast<uint32_t>(Hkv);
  sp.D = static_cast<uint32_t>(D);
  sp.context_len = static_cast<uint32_t>(context_len);
  sp.g = static_cast<uint32_t>(g);
  sp.num_splits = num_splits;
  sp.split_len = split_len;
  sp.scale = scale;
  WGPUBuffer ub_split = graph.make_uniform_buffer(&sp, sizeof(sp));
  BufferBinding split_bindings[5] = {
      {part_o, po_floats * sizeof(float)},
      {part_ml, pml_floats * sizeof(float)},
      {q.buffer, q.nbytes},
      {k_cache.buffer, k_cache.nbytes},
      {v_cache.buffer, v_cache.nbytes}};
  // Compute the thread product in 64-bit + guard before the u32 cast, mirroring
  // the Sdpa.cpp aw_floats > UINT32_MAX guards.
  const uint64_t split_threads = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(num_splits) *
      static_cast<uint64_t>(kSdpaFdSplitWorkgroupSizeX);
  if (split_threads > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU sdpa FlashDecoding: split thread count exceeds uint32 max");
  }
  const uint32_t wgc_split = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(split_threads),
      kSdpaFdSplitWorkgroupSizeX,
      "fd_split");
  const char* split_shader = kSdpaFdSplitWGSL;
  if (graph.kv_f16()) {
    split_shader = kSdpaFdSplitHalfWGSL;
  }
  build_dispatch(
      graph,
      split_shader,
      split_bindings,
      5,
      2,
      ub_split,
      sizeof(sp),
      wgc_split,
      "fd_split");

  // Pass 2: reduce (Hq WGs) -> reads part_o, part_ml; writes out.
  FdReduceParams rp = {};
  rp.D = static_cast<uint32_t>(D);
  rp.num_splits = num_splits;
  WGPUBuffer ub_reduce = graph.make_uniform_buffer(&rp, sizeof(rp));
  BufferBinding reduce_bindings[3] = {
      {out.buffer, out.nbytes},
      {part_o, po_floats * sizeof(float)},
      {part_ml, pml_floats * sizeof(float)}};
  const uint32_t wgc_reduce = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(Hq) * kSdpaFdReduceWorkgroupSizeX,
      kSdpaFdReduceWorkgroupSizeX,
      "fd_reduce");
  build_dispatch(
      graph,
      kSdpaFdReduceWGSL,
      reduce_bindings,
      3,
      1,
      ub_reduce,
      sizeof(rp),
      wgc_reduce,
      "fd_reduce");
}

} // namespace executorch::backends::webgpu
