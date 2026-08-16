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
#include <executorch/backends/webgpu/runtime/ops/to_copy/to_copy.h>
#include <executorch/backends/webgpu/runtime/ops/to_copy/to_copy_bool_to_float_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/to_copy/to_copy_float_to_int_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/to_copy/to_copy_int_to_float_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy.h>

#include <webgpu/webgpu.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct; 16-byte aligned.
struct ConvertParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

// Elementwise int32<->fp32 convert; mirrors Vulkan add_view_copy_convert_node.
void add_convert_op(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    const char* wgsl_source,
    uint32_t wg_size_x,
    const char* op_name) {
  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error(std::string(op_name) + ": null buffer binding");
  }

  // 32-bit only: int64 consts are downcast to int32 by the Vulkan serializer.
  if (in_tensor.elem_size != 4 || out_tensor.elem_size != 4) {
    throw std::runtime_error(
        std::string(op_name) + ": only 32-bit int<->float convert supported");
  }
  if (in_tensor.nbytes != out_tensor.nbytes) {
    throw std::runtime_error(std::string(op_name) + ": numel mismatch");
  }

  uint32_t num_elements = static_cast<uint32_t>(out_tensor.nbytes / 4);

  uint32_t wg_size = utils::clamp_workgroup_size(device, wg_size_x);
  uint32_t workgroup_count =
      utils::compute_1d_workgroup_count(device, num_elements, wg_size, op_name);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ConvertParams params = {};
  params.num_elements = num_elements;

  WGPUBuffer uniform_buffer = graph.create_params_buffer(params);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      wgsl_source,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ConvertParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx =
      graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

  // Dynamic shapes: recompute num_elements/dispatch for the live shape.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const uint64_t numel = utils::numel_of(d);
        g.set_cur_dims(out_id, d);
        ConvertParams p = {};
        p.num_elements = static_cast<uint32_t>(numel);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(),
                static_cast<uint32_t>(numel),
                wg_size,
                "to_copy(resize)");
      });
}

// Decode byte-packed bool storage into numeric fp32 values.
void add_bool_to_float_op(WebGPUGraph& graph, int in_id, int out_id) {
  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("to_copy_bool_to_float: null buffer binding");
  }
  if (!in_tensor.is_bool || in_tensor.elem_size != 1 || out_tensor.is_int ||
      out_tensor.elem_size != sizeof(float) ||
      out_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes / sizeof(float) != in_tensor.nbytes) {
    throw std::runtime_error("to_copy_bool_to_float: dtype/numel mismatch");
  }
  if (in_tensor.nbytes == 0u ||
      in_tensor.nbytes > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "to_copy_bool_to_float: numel must be nonzero and fit u32");
  }

  const uint32_t num_elements = static_cast<uint32_t>(in_tensor.nbytes);
  const uint64_t input_bind_size_u64 =
      (static_cast<uint64_t>(in_tensor.nbytes) + 3u) & ~uint64_t(3);
  if (input_bind_size_u64 > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        "to_copy_bool_to_float: input binding size overflows");
  }
  const size_t input_bind_size = static_cast<size_t>(input_bind_size_u64);

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kToCopyBoolToFloatWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, num_elements, wg_size, "to_copy_bool_to_float");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ConvertParams params = {};
  params.num_elements = num_elements;
  WGPUBuffer uniform_buffer = graph.create_params_buffer(params);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kToCopyBoolToFloatWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           input_bind_size},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ConvertParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx =
      graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& dims = g.cur_dims(in_id);
        const uint64_t numel = utils::numel_of(dims);
        if (numel == 0u || numel > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error(
              "to_copy_bool_to_float(resize): invalid numel");
        }
        g.set_cur_dims(out_id, dims);
        ConvertParams p = {};
        p.num_elements = static_cast<uint32_t>(numel);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(),
                static_cast<uint32_t>(numel),
                wg_size,
                "to_copy_bool_to_float(resize)");
      });
}

void to_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten._to_copy.default args: [self, ...kwargs, out]; out = last value id.
  add_to_copy_node(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

void add_to_copy_node(WebGPUGraph& graph, int in_id, int out_id) {
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (in_tensor.is_bool != out_tensor.is_bool && in_tensor.is_int &&
      out_tensor.is_int) {
    throw std::runtime_error(
        "WebGPU to_copy: bool and integer conversions are unsupported");
  }

  // Same is_int+width = flat byte copy; unique dtype key in the 32-bit domain.
  if (in_tensor.is_int == out_tensor.is_int &&
      in_tensor.elem_size == out_tensor.elem_size) {
    add_flat_copy(graph, in_id, out_id);
    return;
  }

  // int<->float = numeric convert (mirrors Vulkan add_view_copy_convert_node).
  if (in_tensor.is_bool && !out_tensor.is_int &&
      out_tensor.elem_size == sizeof(float)) {
    add_bool_to_float_op(graph, in_id, out_id);
  } else if (in_tensor.is_int && !out_tensor.is_int) {
    add_convert_op(
        graph,
        in_id,
        out_id,
        kToCopyIntToFloatWGSL,
        kToCopyIntToFloatWorkgroupSizeX,
        "to_copy_int_to_float");
  } else {
    add_convert_op(
        graph,
        in_id,
        out_id,
        kToCopyFloatToIntWGSL,
        kToCopyFloatToIntWorkgroupSizeX,
        "to_copy_float_to_int");
  }
}

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten._to_copy.default, to_copy_impl);
}

} // namespace executorch::backends::webgpu
