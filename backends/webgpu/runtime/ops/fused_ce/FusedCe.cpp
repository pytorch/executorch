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
#include <executorch/backends/webgpu/runtime/ops/fused_ce/fused_ce_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/reduce/reduce_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the fused_ce.wgsl Params struct (16-byte aligned).
struct FusedCeParams {
  uint32_t vocab;
  uint32_t n_rows;
  float n_valid;
  float _pad0;
};
static_assert(sizeof(FusedCeParams) == 16, "FusedCeParams must be 16 bytes");

// Mirror reduce.wgsl Params (file-local in Reduce.cpp; re-declared here).
struct ReduceParams {
  uint32_t outer;
  uint32_t r;
  uint32_t inner;
  uint32_t is_mean;
};
static_assert(sizeof(ReduceParams) == 16, "ReduceParams must be 16 bytes");

WGPUBuffer create_uniform(
    WebGPUGraph& graph,
    WGPUDevice device,
    const void* data,
    size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
  std::memcpy(wgpuBufferGetMappedRange(buffer, 0, size), data, size);
  wgpuBufferUnmap(buffer);
  graph.add_uniform_buffer_bytes(size);
  return buffer;
}

// out valuelist packs the 2-tuple (loss, dlogits) as one id.
void fused_ce_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int logits_id = args.at(0);
  const int labels_id = args.at(1);
  const int n_valid_id = args.at(2);
  const std::vector<int>& outs = graph.get_value_list(args.at(3));
  if (outs.size() != 2) {
    throw std::runtime_error(
        "WebGPU fused_ce: expected 2 outputs (loss, dlogits)");
  }
  const int loss_id = outs.at(0);
  const int dlogits_id = outs.at(1);

  WGPUDevice device = graph.device();
  const auto& logits = graph.get_tensor(logits_id);
  const auto& labels = graph.get_tensor(labels_id);
  const auto& dlogits = graph.get_tensor(dlogits_id);
  const auto& loss = graph.get_tensor(loss_id);

  if (logits.dims.size() != 2) {
    throw std::runtime_error("WebGPU fused_ce: logits must be 2D [N, V]");
  }
  const uint64_t n_rows = static_cast<uint64_t>(logits.dims[0]);
  const uint64_t vocab = static_cast<uint64_t>(logits.dims[1]);
  const uint64_t numel = n_rows * vocab;

  if (dlogits.dims != logits.dims) {
    throw std::runtime_error(
        "WebGPU fused_ce: dlogits shape must match logits");
  }
  if (logits.nbytes != numel * sizeof(float) ||
      dlogits.nbytes != numel * sizeof(float)) {
    throw std::runtime_error("WebGPU fused_ce: logits/dlogits fp32-only");
  }
  if (labels.nbytes != n_rows * sizeof(int32_t)) {
    throw std::runtime_error("WebGPU fused_ce: labels must be int32 [N]");
  }
  if (loss.nbytes != sizeof(float)) {
    throw std::runtime_error("WebGPU fused_ce: loss must be a scalar [1]");
  }
  if (graph.get_value_type(n_valid_id) != WebGPUGraph::ValueType::Double) {
    throw std::runtime_error("WebGPU fused_ce: n_valid must be a float scalar");
  }
  const double n_valid = graph.get_double(n_valid_id);
  if (n_valid <= 0.0) {
    throw std::runtime_error("WebGPU fused_ce: n_valid must be positive");
  }
  if (n_rows > utils::queried_max_workgroups(device)) {
    throw std::runtime_error("WebGPU fused_ce: n_rows exceeds dispatch limit");
  }

  WGPUBuffer loss_partial = graph.create_scratch_buffer(n_rows * sizeof(float));

  // one workgroup per row
  FusedCeParams ce_params = {};
  ce_params.vocab = static_cast<uint32_t>(vocab);
  ce_params.n_rows = static_cast<uint32_t>(n_rows);
  ce_params.n_valid = static_cast<float>(n_valid);
  const uint32_t ce_wg =
      utils::clamp_workgroup_size(device, kFusedCeWorkgroupSizeX);
  WGPUBuffer ce_uniform =
      create_uniform(graph, device, &ce_params, sizeof(ce_params));
  WGPUConstantEntry ce_wg_const = {};
  ce_wg_const.key = {"wg_size", WGPU_STRLEN};
  ce_wg_const.value = static_cast<double>(ce_wg);

  utils::ComputePipelineBundle ce_bundle = utils::make_compute_pipeline(
      device,
      kFusedCeWGSL,
      {
          {0, WGPUBufferBindingType_Storage, dlogits.buffer, dlogits.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           logits.buffer,
           logits.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           labels.buffer,
           labels.nbytes},
          {3,
           WGPUBufferBindingType_Storage,
           loss_partial,
           n_rows * sizeof(float)},
          {4, WGPUBufferBindingType_Uniform, ce_uniform, sizeof(ce_params)},
      },
      &ce_wg_const,
      1);

  graph.add_dispatch(
      {ce_bundle.pipeline,
       ce_bundle.bind_group,
       static_cast<uint32_t>(n_rows),
       "fused_ce"});

  wgpuBufferRelease(ce_uniform);

  // reduce loss_partial[N] -> loss[1] (reuses reduce.wgsl)
  ReduceParams r_params = {};
  r_params.outer = 1u;
  r_params.r = static_cast<uint32_t>(n_rows);
  r_params.inner = 1u;
  r_params.is_mean = 0u; // mean already folded into loss_partial
  const uint32_t r_wg =
      utils::clamp_workgroup_size(device, kReduceWorkgroupSizeX);
  const uint32_t r_wgc =
      utils::compute_1d_workgroup_count(device, 1u, r_wg, "fused_ce_reduce");
  WGPUBuffer r_uniform =
      create_uniform(graph, device, &r_params, sizeof(r_params));
  WGPUConstantEntry r_wg_const = {};
  r_wg_const.key = {"wg_size", WGPU_STRLEN};
  r_wg_const.value = static_cast<double>(r_wg);

  utils::ComputePipelineBundle r_bundle = utils::make_compute_pipeline(
      device,
      kReduceWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           loss_partial,
           n_rows * sizeof(float)},
          {1, WGPUBufferBindingType_Storage, loss.buffer, loss.nbytes},
          {2, WGPUBufferBindingType_Uniform, r_uniform, sizeof(r_params)},
      },
      &r_wg_const,
      1);

  graph.add_dispatch(
      {r_bundle.pipeline, r_bundle.bind_group, r_wgc, "fused_ce_reduce"});

  wgpuBufferRelease(r_uniform);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.fused_ce.default, fused_ce_impl);
}

} // namespace executorch::backends::webgpu
