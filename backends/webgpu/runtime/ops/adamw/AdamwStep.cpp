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
#include <executorch/backends/webgpu/runtime/ops/adamw/adamw_step_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct AdamwStepParams {
  uint32_t numel;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bias_correction1;
  float bias_correction2;
  float _pad3;
};
static_assert(sizeof(AdamwStepParams) == 48, "params must be 48 bytes");

// AdamW step over an fp32 latent (elementwise, in place); mirrors torch AdamW.
void adamw_step_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int param_id = args.at(0);
  const int m_id = args.at(1);
  const int v_id = args.at(2);
  const int grad_id = args.at(3);

  WGPUDevice device = graph.device();
  const auto& param = graph.get_tensor(param_id);
  const auto& m = graph.get_tensor(m_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& grad = graph.get_tensor(grad_id);

  uint64_t numel = 1;
  for (int64_t d : param.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (param.dims.empty() || numel == 0) {
    throw std::runtime_error("adamw_step: empty param");
  }
  const uint64_t bytes = numel * sizeof(float);
  if (param.nbytes != bytes || m.nbytes != bytes || v.nbytes != bytes ||
      grad.nbytes != bytes) {
    throw std::runtime_error(
        "adamw_step: param/m/v/grad must be fp32 and same numel");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("adamw_step: numel exceeds u32");
  }

  auto scalar = [&](int id, const char* name) -> float {
    if (graph.get_value_type(id) != WebGPUGraph::ValueType::Double) {
      throw std::runtime_error(
          std::string("adamw_step: ") + name + " must be a float scalar");
    }
    return static_cast<float>(graph.get_double(id));
  };

  AdamwStepParams params = {};
  params.numel = static_cast<uint32_t>(numel);
  params.lr = scalar(args.at(4), "lr");
  params.beta1 = scalar(args.at(5), "beta1");
  params.beta2 = scalar(args.at(6), "beta2");
  params.eps = scalar(args.at(7), "eps");
  params.weight_decay = scalar(args.at(8), "weight_decay");
  params.bias_correction1 = scalar(args.at(9), "bias_correction1");
  params.bias_correction2 = scalar(args.at(10), "bias_correction2");
  if (params.bias_correction1 == 0.0f || params.bias_correction2 == 0.0f) {
    throw std::runtime_error("adamw_step: bias corrections must be non-zero");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kAdamwStepWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, params.numel, wg_size, "adamw_step");

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(params));
  graph.add_uniform_buffer_bytes(sizeof(params));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kAdamwStepWGSL,
      {
          {0, WGPUBufferBindingType_Storage, param.buffer, param.nbytes},
          {1, WGPUBufferBindingType_Storage, m.buffer, m.nbytes},
          {2, WGPUBufferBindingType_Storage, v.buffer, v.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, grad.buffer, grad.nbytes},
          {4, WGPUBufferBindingType_Uniform, uniform_buffer, sizeof(params)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline, bundle.bind_group, workgroup_count, "adamw_step"});

  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.adamw_step.default, adamw_step_impl);
}

} // namespace executorch::backends::webgpu
