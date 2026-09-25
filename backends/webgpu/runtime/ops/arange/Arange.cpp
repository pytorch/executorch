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
#include <executorch/backends/webgpu/runtime/ops/arange/arange_int_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/arange/arange_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout shared by both shaders (16 bytes). start/step are
// reinterpreted as i32 by the integer variant, so keep the slots the same
// width.
struct ArangeParams {
  uint32_t num_elements;
  union {
    float f;
    int32_t i;
  } start;
  union {
    float f;
    int32_t i;
  } step;
  uint32_t _pad;
};
static_assert(
    sizeof(ArangeParams) % 16u == 0u,
    "WebGPU parameter blocks must have a 16-byte-aligned size");

// start/end/step arrive as static Int or Double scalars, or as Null when the
// caller omitted them, in which case aten's defaults apply. A SymInt would make
// the output length dynamic, which the fixed dispatch below cannot honor.
// if_omitted == nullptr marks an argument that has no default and must be
// present.
double read_scalar(
    WebGPUGraph& graph,
    int id,
    const char* what,
    const double* if_omitted) {
  switch (graph.get_value_type(id)) {
    case WebGPUGraph::ValueType::Int:
      return static_cast<double>(graph.get_int(id));
    case WebGPUGraph::ValueType::Double:
      return graph.get_double(id);
    case WebGPUGraph::ValueType::Null:
      if (if_omitted == nullptr) {
        throw std::runtime_error(
            std::string("arange: ") + what + " is required");
      }
      return *if_omitted;
    default:
      throw std::runtime_error(
          std::string("arange: dynamic/unsupported ") + what);
  }
}

// aten.arange.start_step(start, end, step, *, dtype, layout, device,
// pin_memory) args: [start, end, step, ..., out]. Writes start + i * step over
// the pre-allocated output; the element count comes from the output tensor
// rather than from (end - start) / step so it stays consistent with memory
// planning.
void arange_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 4) {
    throw std::runtime_error("arange: expected at least 4 args");
  }
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();
  const auto& out_tensor = graph.get_tensor(out_id);
  if (out_tensor.buffer == nullptr) {
    throw std::runtime_error("arange: null output buffer");
  }
  // Two 4-byte variants: fp32 and i32. A bool output is 1-byte packed and has
  // no meaningful arange, so reject it rather than mis-stride the buffer.
  if (out_tensor.is_bool || out_tensor.elem_size != 4u) {
    throw std::runtime_error("arange: only 4-byte fp32 or int32 output");
  }
  const bool is_int = out_tensor.is_int;

  constexpr double kStartDefault = 0.0;
  constexpr double kStepDefault = 1.0;
  const double start = read_scalar(graph, args.at(0), "start", &kStartDefault);
  const double step = read_scalar(graph, args.at(2), "step", &kStepDefault);
  // end does not feed the dispatch (the length comes from the output tensor),
  // but it must still be static: a SymInt end changes the live length, and
  // nothing here recomputes the output dims the way Vulkan's resize does, so a
  // dynamic end would silently keep emitting the max-sized output.
  (void)read_scalar(graph, args.at(1), "end", nullptr);
  if (step == 0.0) {
    throw std::runtime_error("arange: step must be non-zero");
  }

  const uint64_t numel = out_tensor.nbytes / 4u;
  if (numel == 0 || numel > UINT32_MAX) {
    throw std::runtime_error("arange: output numel is zero or exceeds u32");
  }
  const uint32_t num_elements = static_cast<uint32_t>(numel);

  uint32_t wg_size = utils::clamp_workgroup_size(device, kArangeWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_elements, wg_size, "arange");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ArangeParams params = {};
  params.num_elements = num_elements;
  if (is_int) {
    params.start.i = static_cast<int32_t>(start);
    params.step.i = static_cast<int32_t>(step);
  } else {
    params.start.f = static_cast<float>(start);
    params.step.f = static_cast<float>(step);
  }
  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ArangeParams));
  graph.add_uniform_buffer_bytes(sizeof(ArangeParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      is_int ? kArangeIntWGSL : kArangeWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ArangeParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "",
       workgroup_count.y});

  // Dynamic shapes: recompute num_elements/dispatch from the live output dims.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      out_id,
      [out_id, params, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(out_id);
        const uint64_t n = utils::numel_of(d);
        ArangeParams p = params;
        p.num_elements = static_cast<uint32_t>(n);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), static_cast<uint32_t>(n), wg_size, "arange(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      });

  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.arange.start_step, arange_impl);
}

} // namespace executorch::backends::webgpu
