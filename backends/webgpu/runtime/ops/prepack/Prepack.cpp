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

// Materialize a constant to its GPU buffer: a dtype-agnostic byte copy.
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
  if (src.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("WebGPU prepack: null buffer binding");
  }

  graph.add_prepack_copy(src.buffer, out.buffer, out.nbytes);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.prepack.default, prepack_impl);
}

} // namespace executorch::backends::webgpu
