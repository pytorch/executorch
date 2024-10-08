/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNStatus.h>
#include <executorch/backends/xnnpack/runtime/profiling/XNNProfiler.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <xnnpack.h>
#include <map>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};

  profiling::XNNProfiler profiler_;
  std::vector<uint32_t> input_ids_;
  std::vector<uint32_t> output_ids_;
  std::vector<xnn_external_value> externals_;

 public:
  XNNExecutor() = default;

  inline size_t getNumInputs() {
    return input_ids_.size();
  }

  inline size_t getNumOutputs() {
    return output_ids_.size();
  }

  /**
   * Initialize the XNNExecutor with a given runtime and input/output ids.
   * The input/output ids are expected to be sorted in order of their
   * flatbuffer id_outs
   */
  ET_NODISCARD executorch::runtime::Error initialize(
      xnn_runtime_t runtime,
      std::vector<uint32_t>&& input_ids,
      std::vector<uint32_t>&& output_ids);

  /**
   * Prepares the arguments for runtime graph execution.
   * args is an array of EValues that will be passed into the runtime.
   * input shapes will be propagated through the runtime, and perform
   * any additional memory planning as needed
   */
  ET_NODISCARD executorch::runtime::Error prepare_args(
      executorch::runtime::EValue** args);

  /**
   * Executes the graph using the args prepared at prepare_args().
   */
  ET_NODISCARD executorch::runtime::Error forward(
      executorch::runtime::BackendExecutionContext& context);

  /**
   * Prepares the outputs to be returned by the delegate
   *
   * Performs any post processing of outputs like tensor resizing
   */
  ET_NODISCARD executorch::runtime::Error resize_outputs(
      executorch::runtime::EValue** args) const;

  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
