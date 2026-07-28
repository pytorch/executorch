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
#include <executorch/backends/webgpu/runtime/ops/layer_norm/native_layer_norm_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct LayerNormParams {
  uint32_t num_rows;
  uint32_t row_width;
  float epsilon;
  uint32_t has_affine;
};
static_assert(
    sizeof(LayerNormParams) == 16,
    "LayerNormParams must be 16 bytes");

// aten.native_layer_norm.default args: [in, normalized_shape, weight, bias,
// eps, out] where out is a ValueList [out, mean, rstd] (mirrors Vulkan
// NativeLayerNorm.cpp). normalized_shape (args[1]) only constrains the last
// dim.
void native_layer_norm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(2);
  const int bias_id = args.at(3);
  const int eps_id = args.at(4);
  const int out_list_id = args.at(5);

  if (graph.get_value_type(out_list_id) != WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error(
        "WebGPU native_layer_norm: out is not a ValueList");
  }
  const std::vector<int>& outs = graph.get_value_list(out_list_id);
  if (outs.size() != 3) {
    throw std::runtime_error(
        "WebGPU native_layer_norm: expected 3 outputs (out, mean, rstd)");
  }
  const int out_id = outs.at(0);
  const int mean_id = outs.at(1);
  const int rstd_id = outs.at(2);

  WGPUDevice device = graph.device();

  float epsilon =
      utils::scalar_or(graph, eps_id, std::numeric_limits<float>::epsilon());

  const auto& in_tensor = graph.get_tensor(in_id);
  if (in_tensor.dims.empty() || in_tensor.nbytes == 0) {
    throw std::runtime_error("WebGPU native_layer_norm: empty input");
  }
  const uint32_t row_width = static_cast<uint32_t>(in_tensor.dims.back());
  if (row_width == 0) {
    throw std::runtime_error("WebGPU native_layer_norm: zero row width");
  }
  // The shader views t_in/t_out/t_weight/t_bias as vec4<f32> over row_width;
  // every model in scope (DaViT/BART/Whisper/Voxtral hidden dims 768/1024/
  // 1280/3072) is always %4==0.
  if (row_width % 4 != 0) {
    throw std::runtime_error(
        "WebGPU native_layer_norm: row_width must be a multiple of 4");
  }
  const uint64_t in_numel =
      utils::check_fp32(in_tensor, "native_layer_norm", "input");
  const uint32_t num_rows = static_cast<uint32_t>(in_numel / row_width);
  if (num_rows == 0) {
    throw std::runtime_error("WebGPU native_layer_norm: zero rows");
  }

  // Near-square 2D grid of workgroups (1 workgroup = 1 row) past the 65535
  // per-dimension ceiling; stride_x lets the shader decode a flat row index
  // as workgroup_id.y * stride_x + workgroup_id.x.
  utils::DispatchGrid grid =
      utils::compute_row_dispatch_grid(device, num_rows, "native_layer_norm");

  // weight/bias are optional: aten.native_layer_norm passes None for both when
  // there is no affine (e.g. the group_norm LN-reframe). When absent, bind
  // dummy storage buffers and gate the affine in the shader (has_affine == 0).
  const bool has_affine =
      graph.get_value_type(weight_id) == WebGPUGraph::ValueType::Tensor &&
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  LayerNormParams params = {};
  params.num_rows = num_rows;
  params.row_width = row_width;
  params.epsilon = epsilon;
  params.has_affine = has_affine ? 1u : 0u;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(LayerNormParams));
  graph.add_uniform_buffer_bytes(sizeof(LayerNormParams));

  const auto& out_tensor = graph.get_tensor(out_id);
  const auto& mean_tensor = graph.get_tensor(mean_id);
  const auto& rstd_tensor = graph.get_tensor(rstd_id);

  utils::OptionalBinding weight = utils::make_optional_binding(
      device,
      has_affine,
      has_affine ? graph.get_tensor(weight_id).buffer : nullptr,
      has_affine ? graph.get_tensor(weight_id).nbytes : 0);
  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_affine,
      has_affine ? graph.get_tensor(bias_id).buffer : nullptr,
      has_affine ? graph.get_tensor(bias_id).nbytes : 0);

  WGPUConstantEntry stride_const = {};
  stride_const.key = {"stride_x", WGPU_STRLEN};
  stride_const.value = static_cast<double>(grid.stride_x);

  // out(rw,0), in(ro,1), weight(ro,2), bias(ro,3), mean(rw,4), rstd(rw,5),
  // params(uniform,6).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kNativeLayerNormWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight.buffer,
           weight.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, bias.buffer, bias.nbytes},
          {4,
           WGPUBufferBindingType_Storage,
           mean_tensor.buffer,
           mean_tensor.nbytes},
          {5,
           WGPUBufferBindingType_Storage,
           rstd_tensor.buffer,
           rstd_tensor.nbytes},
          {6,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(LayerNormParams)},
      },
      &stride_const,
      1);

  static_assert(
      kNativeLayerNormWorkgroupSizeX == 64,
      "must match @workgroup_size and WG_SIZE in native_layer_norm.wgsl");
  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, grid.count_x, grid.count_y);

  wgpuBufferRelease(uniform_buffer);
  if (weight.owned_dummy != nullptr) {
    wgpuBufferRelease(weight.owned_dummy);
  }
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.native_layer_norm.default, native_layer_norm_impl);
}

} // namespace executorch::backends::webgpu
