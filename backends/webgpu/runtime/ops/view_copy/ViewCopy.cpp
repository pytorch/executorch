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
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy.h>

#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

void add_flat_copy(WebGPUGraph& graph, int in_id, int out_id) {
  // get_tensor doesn't type-check; assert both args are tensors (fail loud).
  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("flat_copy: in/out arg is not a tensor");
  }

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  // Contiguous reshape = flat byte copy; mirrors Vulkan view_buffer (no-remap).

  // 4-byte alignment guard (fp32 element size); does not verify dtype.
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("flat_copy: operand not 4-byte aligned");
  }

  // view preserves numel; this guard also prevents an OOB copy.
  if (in_tensor.nbytes != out_tensor.nbytes) {
    throw std::runtime_error("flat_copy: input/output size mismatch");
  }

  // Aliased in/out already in place; CopyBufferToBuffer rejects src == dst.
  const bool aliased = in_tensor.buffer == out_tensor.buffer;
  if (!aliased) {
    graph.add_buffer_copy(
        in_tensor.buffer, out_tensor.buffer, out_tensor.nbytes);
  }
  const size_t dispatch_idx = aliased ? 0 : graph.num_dispatches() - 1;

  // Dynamic shapes: view preserves numel; copy_nbytes + out dims track live in.
  std::vector<int64_t> out_max = out_tensor.dims;
  graph.add_tensor_resize_hook(
      in_id, [in_id, out_id, out_max, dispatch_idx, aliased](WebGPUGraph& g) {
        const uint64_t target = utils::numel_of(g.cur_dims(in_id));
        std::vector<int64_t> od = out_max;
        const uint64_t maxnumel = utils::numel_of(out_max);
        if (maxnumel != target) {
          for (size_t d = 0; d < od.size(); d++) {
            const uint64_t rest = maxnumel / static_cast<uint64_t>(out_max[d]);
            if (rest != 0 && target % rest == 0) {
              const uint64_t nd = target / rest;
              if (nd <= static_cast<uint64_t>(out_max[d])) {
                od[d] = static_cast<int64_t>(nd);
                break;
              }
            }
          }
        }
        g.set_cur_dims(out_id, od);
        if (!aliased) {
          g.dispatch_at(dispatch_idx).copy_nbytes =
              static_cast<size_t>(target) * sizeof(float);
        }
      });
}

namespace {

// view_copy = contiguous reshape = flat copy (mirrors Vulkan view_buffer).
void view_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, size, out]; out = last value-id (shape from out_tensor.dims).
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

// clone = flat copy; survives Vulkan RemoveRedundantOpsTransform in Llama 1B.
void clone_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, memory_format?, out]; out = last value-id.
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.view_copy.default, view_copy_impl);
  WEBGPU_REGISTER_OP(aten.clone.default, clone_impl);
}

} // namespace executorch::backends::webgpu
