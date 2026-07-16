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
#include <executorch/backends/webgpu/runtime/ops/batch_norm/batch_norm_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct BNParams {
  uint32_t num_elements;
  uint32_t C;
  uint32_t HW;
  uint32_t has_weight;
  uint32_t has_bias;
  float eps;
  uint32_t _p0;
  uint32_t _p1;
};
static_assert(sizeof(BNParams) == 32, "BNParams must be 32 bytes");

// _native_batch_norm_legit_no_training: per-channel affine from running stats.
void batch_norm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 8) {
    throw std::runtime_error("WebGPU batch_norm: expected >=8 args");
  }
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int mean_id = args.at(3);
  const int var_id = args.at(4);
  const double eps = graph.get_double(args.at(6));
  const std::vector<int>& outs = graph.get_value_list(args.at(args.size() - 1));
  if (outs.empty()) {
    throw std::runtime_error("WebGPU batch_norm: empty out ValueList");
  }
  const int out_id = outs.at(0);

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);
  const auto& mean = graph.get_tensor(mean_id);
  const auto& var = graph.get_tensor(var_id);

  if (in.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error("WebGPU batch_norm: expected 4D in/out");
  }
  const uint32_t N = static_cast<uint32_t>(in.dims[0]);
  const uint32_t C = static_cast<uint32_t>(in.dims[1]);
  const uint32_t HW =
      static_cast<uint32_t>(in.dims[2]) * static_cast<uint32_t>(in.dims[3]);

  uint64_t out_numel = uint64_t(N) * C * HW;
  if (out.nbytes != out_numel * sizeof(float)) {
    throw std::runtime_error("WebGPU batch_norm: fp32-only (byte mismatch)");
  }
  if (out_numel > 0xFFFFFFFFull) {
    throw std::runtime_error("WebGPU batch_norm: output too large");
  }

  const bool has_weight =
      graph.get_value_type(weight_id) == WebGPUGraph::ValueType::Tensor;
  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      static_cast<uint32_t>(out_numel),
      kBatchNormWorkgroupSizeX,
      "batch_norm");

  BNParams params = {};
  params.num_elements = static_cast<uint32_t>(out_numel);
  params.C = C;
  params.HW = HW;
  params.has_weight = has_weight ? 1u : 0u;
  params.has_bias = has_bias ? 1u : 0u;
  params.eps = static_cast<float>(eps);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(BNParams));
  graph.add_uniform_buffer_bytes(sizeof(BNParams));

  utils::OptionalBinding wbind = utils::make_optional_binding(
      device,
      has_weight,
      has_weight ? graph.get_tensor(weight_id).buffer : nullptr,
      has_weight ? graph.get_tensor(weight_id).nbytes : 0);
  utils::OptionalBinding bbind = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);

  auto constants = utils::make_grid_constants(grid);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kBatchNormWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           wbind.buffer,
           wbind.nbytes},
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           bbind.buffer,
           bbind.nbytes},
          {4, WGPUBufferBindingType_ReadOnlyStorage, mean.buffer, mean.nbytes},
          {5, WGPUBufferBindingType_ReadOnlyStorage, var.buffer, var.nbytes},
          {6, WGPUBufferBindingType_Uniform, uniform_buffer, sizeof(BNParams)},
      },
      constants.data(),
      constants.size());

  WebGPUDispatch dispatch{};
  dispatch.pipeline = bundle.pipeline;
  dispatch.bind_group = bundle.bind_group;
  dispatch.workgroup_count_x = grid.count_x;
  dispatch.workgroup_count_y = grid.count_y;
  graph.add_dispatch(dispatch);

  wgpuBufferRelease(uniform_buffer);
  if (wbind.owned_dummy != nullptr) {
    wgpuBufferRelease(wbind.owned_dummy);
  }
  if (bbind.owned_dummy != nullptr) {
    wgpuBufferRelease(bbind.owned_dummy);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      aten._native_batch_norm_legit_no_training.default, batch_norm_impl);
}

} // namespace executorch::backends::webgpu
