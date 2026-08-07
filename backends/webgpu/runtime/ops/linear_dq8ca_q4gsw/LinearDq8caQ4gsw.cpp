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
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/QuantizedLinear.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/choose_qparams_dq8ca_fused_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/quantize_dequantize_per_row_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct QuantizeDequantizeParams {
  uint32_t num_elements;
  uint32_t num_rows;
  uint32_t row_width;
  uint32_t _pad;
};
static_assert(sizeof(QuantizeDequantizeParams) == 16);
static_assert(
    kChooseQparamsDq8caFusedWorkgroupSizeX == utils::kCqpQdqFusedInvocations);

struct QuantizeDequantizeState {
  QuantizeDequantizeParams params;
  utils::WgCount grid;
};

QuantizeDequantizeState make_quantize_dequantize_state(
    WGPUDevice device,
    const std::vector<int64_t>& input_dims,
    uint32_t max_rows,
    uint32_t row_width,
    uint32_t workgroup_size) {
  if (input_dims.empty() || input_dims.back() != row_width) {
    throw std::runtime_error(
        "WebGPU linear_dq8ca_q4gsw: live row width mismatch");
  }
  const uint64_t numel = utils::numel_of(input_dims);
  if (numel == 0u || numel % row_width != 0u || numel > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU linear_dq8ca_q4gsw: invalid live input numel");
  }
  const uint64_t rows = numel / row_width;
  if (rows == 0u || rows > max_rows) {
    throw std::runtime_error(
        "WebGPU linear_dq8ca_q4gsw: live rows exceed the build-time max");
  }

  QuantizeDequantizeState state = {};
  state.params = {
      static_cast<uint32_t>(numel), static_cast<uint32_t>(rows), row_width, 0u};
  state.grid = utils::compute_2d_workgroup_count(
      device,
      state.params.num_elements,
      workgroup_size,
      "linear_dq8ca_q4gsw_qdq");
  return state;
}

utils::WgCount make_cqp_fused_grid(
    WGPUDevice device,
    const QuantizeDequantizeParams& params) {
  const uint32_t grid_x = utils::clamp_workgroup_count(device, params.num_rows);
  if (grid_x == 0u) {
    throw std::runtime_error("WebGPU linear_dq8ca_q4gsw(fused): zero dispatch");
  }
  return {grid_x, 1u};
}

WebGPUGraph::CqpFusionSite claim_cqp_fusion_site(
    WebGPUGraph& graph,
    int input_id,
    int input_scales_id,
    int input_zero_points_id,
    uint32_t rows,
    uint32_t row_width,
    uint64_t numel) {
  WebGPUGraph::CqpFusionSite site = graph.claim_cqp_fusion_site(
      input_id, input_scales_id, input_zero_points_id, rows, row_width);
  if (!site.valid) {
    return site;
  }
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(graph.device(), &limits) != WGPUStatus_Success ||
      !utils::is_cqp_qdq_fusion_eligible(
          rows,
          row_width,
          numel,
          site.quant_min,
          site.quant_max,
          true,
          true,
          false,
          limits.maxComputeInvocationsPerWorkgroup,
          limits.maxComputeWorkgroupSizeX,
          limits.maxComputeWorkgroupStorageSize)) {
    return WebGPUGraph::CqpFusionSite{};
  }
  return site;
}

void linear_dq8ca_q4gsw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int input_id = args.at(0);
  const int input_scales_id = args.at(1);
  const int input_zero_points_id = args.at(2);
  const int weight_id = args.at(3);
  const int weight_scales_id = args.at(5);
  const int group_size_id = args.at(6);
  const int bias_id = args.at(7);
  const int output_id = args.at(8);

  const auto& input = graph.get_tensor(input_id);
  const auto& input_scales = graph.get_tensor(input_scales_id);
  const auto& input_zero_points = graph.get_tensor(input_zero_points_id);
  if (input.buffer == nullptr || input.dims.empty() || input.is_int ||
      !utils::is_fp32_tensor(input)) {
    throw std::runtime_error("WebGPU linear_dq8ca_q4gsw: expected fp32 input");
  }
  if (input.dims.back() <= 0 ||
      static_cast<uint64_t>(input.dims.back()) > UINT32_MAX) {
    throw std::runtime_error("WebGPU linear_dq8ca_q4gsw: invalid row width");
  }
  const uint32_t row_width = static_cast<uint32_t>(input.dims.back());
  const uint64_t input_numel = utils::numel_of(input.dims);
  if (row_width == 0u || input_numel % row_width != 0u) {
    throw std::runtime_error("WebGPU linear_dq8ca_q4gsw: invalid input shape");
  }
  const uint64_t max_rows64 = input_numel / row_width;
  if (max_rows64 == 0u || max_rows64 > UINT32_MAX) {
    throw std::runtime_error("WebGPU linear_dq8ca_q4gsw: rows out of range");
  }
  const uint32_t max_rows = static_cast<uint32_t>(max_rows64);
  if (input_scales.buffer == nullptr || input_zero_points.buffer == nullptr ||
      input_scales.is_int || input_scales.elem_size != sizeof(float) ||
      !input_zero_points.is_int8 ||
      input_zero_points.elem_size != sizeof(int8_t) ||
      input_scales.dims != input_zero_points.dims ||
      utils::numel_of(input_scales.dims) != max_rows ||
      input_scales.nbytes != max_rows * sizeof(float) ||
      input_zero_points.nbytes != max_rows) {
    throw std::runtime_error(
        "WebGPU linear_dq8ca_q4gsw: invalid per-row qparams");
  }

  WebGPUGraph::ScopedScratch scratch(
      &graph, graph.acquire_scratch(input.nbytes));
  const WebGPUGraph::CqpFusionSite fusion = claim_cqp_fusion_site(
      graph,
      input_id,
      input_scales_id,
      input_zero_points_id,
      max_rows,
      row_width,
      input_numel);
  const bool use_fused = fusion.valid;
  const uint32_t workgroup_size = utils::clamp_workgroup_size(
      graph.device(), kQuantizeDequantizePerRowWorkgroupSizeX);
  const QuantizeDequantizeState initial_state = make_quantize_dequantize_state(
      graph.device(), input.dims, max_rows, row_width, workgroup_size);
  const utils::WgCount initial_grid = use_fused
      ? make_cqp_fused_grid(graph.device(), initial_state.params)
      : initial_state.grid;
  WGPUBuffer uniform = graph.make_uniform_buffer(
      &initial_state.params, sizeof(QuantizeDequantizeParams));

  WGPUConstantEntry workgroup_constant = {};
  workgroup_constant.key = {"wg_size", WGPU_STRLEN};
  workgroup_constant.value = static_cast<double>(workgroup_size);
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      graph.device(),
      use_fused ? kChooseQparamsDq8caFusedWGSL : kQuantizeDequantizePerRowWGSL,
      {
          {0, WGPUBufferBindingType_Storage, scratch.buf, input.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           input.buffer,
           input.nbytes},
          {2,
           use_fused ? WGPUBufferBindingType_Storage
                     : WGPUBufferBindingType_ReadOnlyStorage,
           input_scales.buffer,
           input_scales.nbytes},
          {3,
           use_fused ? WGPUBufferBindingType_Storage
                     : WGPUBufferBindingType_ReadOnlyStorage,
           input_zero_points.buffer,
           std::max(((input_zero_points.nbytes + 3u) / 4u) * 4u, size_t(4))},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform,
           sizeof(QuantizeDequantizeParams)},
      },
      use_fused ? nullptr : &workgroup_constant,
      use_fused ? 0u : 1u);

  const size_t dispatch_index = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       initial_grid.x,
       use_fused ? "linear_dq8ca_q4gsw_cqp_fused" : "linear_dq8ca_q4gsw_qdq",
       initial_grid.y});
  graph.add_tensor_resize_hook(
      input_id,
      [input_id,
       max_rows,
       row_width,
       workgroup_size,
       use_fused,
       dispatch_index,
       uniform](WebGPUGraph& g) {
        const QuantizeDequantizeState state = make_quantize_dequantize_state(
            g.device(),
            g.cur_dims(input_id),
            max_rows,
            row_width,
            workgroup_size);
        wgpuQueueWriteBuffer(
            g.queue(), uniform, 0, &state.params, sizeof(state.params));
        const utils::WgCount grid = use_fused
            ? make_cqp_fused_grid(g.device(), state.params)
            : state.grid;
        auto& dispatch = g.dispatch_at(dispatch_index);
        dispatch.workgroup_count_x = grid.x;
        dispatch.workgroup_count_y = grid.y;
      });
  if (use_fused) {
    auto& producer = graph.dispatch_at(fusion.dispatch_index);
    producer.workgroup_count_x = 0u;
    producer.workgroup_count_y = 0u;
    *fusion.producer_elided = true;
  }

  graph.own_uniform_buffer(uniform);
  const std::vector<int> q4_args = {
      input_id, weight_id, weight_scales_id, group_size_id, bias_id, output_id};
  q4gsw_linear_impl_with_input_buffer(
      graph, q4_args, scratch.buf, input.nbytes);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw_impl);
}

} // namespace executorch::backends::webgpu
