/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Materialize a constant into the prepack-output buffer via one CPU->GPU write.
void prepack_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // et_vk.prepack.default args: [src (constant), out].
  if (args.size() != 2) {
    throw std::runtime_error("WebGPU prepack: expected 2 args (src, out)");
  }
  const auto& src = graph.get_tensor(args.at(0));
  const auto& out = graph.get_tensor(args.at(1));

  if (src.dims != out.dims) {
    throw std::runtime_error("WebGPU prepack: src/out shape mismatch");
  }
  if (src.elem_size != out.elem_size) {
    throw std::runtime_error(
        "WebGPU prepack: src/out dtype mismatch (cast unsupported)");
  }
  if (src.nbytes != out.nbytes) {
    throw std::runtime_error("WebGPU prepack: src/out byte-size mismatch");
  }
  if (out.buffer == nullptr) {
    throw std::runtime_error("WebGPU prepack: null out buffer binding");
  }

  // Sole materialization: write the .pte bytes once, straight into the
  // consumer's buffer (no eager src buffer, no buffer->buffer copy).
  // Correctness of this write-once relies on `out` being a dedicated buffer
  // (the partitioner gives prepack outputs mem_obj_id=-1, so it is never
  // memory-plan aliased with a transient that execute() would later overwrite).
  graph.materialize_constant(args.at(0), out.buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.prepack.default, prepack_impl);
}

} // namespace executorch::backends::webgpu
