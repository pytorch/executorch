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

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};

  profiling::XNNProfiler profiler_;
  size_t num_inputs_;
  size_t num_outputs_;
  std::vector<xnn_external_value> externals_;

 public:
  XNNExecutor() = default;

  inline size_t getNumInputs() {
    return num_inputs_;
  }

  inline size_t getNumOutputs() {
    return num_outputs_;
  }

  /**
   * Initialize the XNNExecutor with a given runtime and input/output ids.
   * The input/output ids are expected to be sorted in order of their
   * flatbuffer id_outs
   */
  __ET_NODISCARD Error
  initialize(xnn_runtime_t runtime, uint32_t num_inputs, uint32_t num_outputs);

  /**
   * Prepares the arguments for runtime graph execution.
   * args is an array of EValues that will be passed into the runtime.
   * input shapes will be propagated through the runtime, and perform
   * any additionaly memory planning as needed
   */
  __ET_NODISCARD Error prepare_args(EValue** args);

  /**
   * Executes the graph using the args prepared at prepare_args().
   */
  __ET_NODISCARD Error forward(BackendExecutionContext& context);

  /**
   * Prepares the outputs to be returned by the delegate
   *
   * Performs any post processing of outputs like tensor resizing
   */
  __ET_NODISCARD Error resize_outputs(EValue** args) const;

  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
