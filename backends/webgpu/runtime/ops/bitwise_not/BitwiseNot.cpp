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
#include <executorch/backends/webgpu/runtime/ops/bitwise_not/bitwise_not_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct BitwiseNotParams {
  uint32_t num_words;
  uint32_t _pad[3];
};
static_assert(
    sizeof(BitwiseNotParams) == 16,
    "BitwiseNotParams must be 16 bytes");

// out = ~a on 1-byte bools; mirrors Vulkan bitwise_not (1-X). args: [a, out].
void bitwise_not_op(WebGPUGraph& graph, const std::vector<int>& args) {
  const int a_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(a_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("bitwise_not: a/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& a_tensor = graph.get_tensor(a_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (a_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("bitwise_not: null buffer binding");
  }
  // a/out are 1-byte bool tensors (int-typed, NOT int8-quantized).
  if (!a_tensor.is_int || !out_tensor.is_int || a_tensor.elem_size != 1 ||
      out_tensor.elem_size != 1) {
    throw std::runtime_error("bitwise_not: a/out must be 1-byte bool tensors");
  }
  const uint64_t numel = out_tensor.nbytes;
  // bool packed 4/word (array<u32>); numel%4==0 gates the u32 binding.
  if (numel == 0u || numel % 4u != 0u || numel > UINT32_MAX) {
    throw std::runtime_error("bitwise_not: numel must be a nonzero mult of 4");
  }
  if (a_tensor.nbytes != numel) {
    throw std::runtime_error("bitwise_not: a/out numel mismatch (same-shape)");
  }

  BitwiseNotParams params = {};
  params.num_words = static_cast<uint32_t>(numel / 4u);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kBitwiseNotWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, params.num_words, wg_size, "bitwise_not");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(BitwiseNotParams));
  graph.add_uniform_buffer_bytes(sizeof(BitwiseNotParams));

  // out (rw storage) + a (ro storage) + params (uniform).
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kBitwiseNotWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           a_tensor.buffer,
           a_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(BitwiseNotParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "bitwise_not",
       workgroup_count.y});

  // Dynamic shapes: recompute num_words/dispatch; out follows a (same-shape).
  WGPUBuffer params_buf = uniform_buffer;
  auto resize =
      [a_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(a_id);
        const uint64_t n = utils::numel_of(d);
        if (n == 0u || n % 4u != 0u || n > UINT32_MAX) {
          throw std::runtime_error(
              "bitwise_not(resize): numel must be a mult of 4");
        }
        g.set_cur_dims(out_id, d);
        BitwiseNotParams p = {};
        p.num_words = static_cast<uint32_t>(n / 4u);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.num_words, wg_size, "bitwise_not");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      };
  graph.add_tensor_resize_hook(a_id, resize);

  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.bitwise_not.default, bitwise_not_op);
}

} // namespace executorch::backends::webgpu
