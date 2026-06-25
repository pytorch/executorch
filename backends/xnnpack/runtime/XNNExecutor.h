/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNStatus.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspace.h>
#include <executorch/backends/xnnpack/runtime/profiling/XNNProfiler.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <xnnpack.h>
#include <atomic>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

// Forward-declared to keep XNNWeightsCache.h out of this header.
class XNNWeightsCache;

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};

  profiling::XNNProfiler profiler_;
  std::vector<uint32_t> input_ids_;
  std::vector<uint32_t> output_ids_;
  std::vector<xnn_external_value> externals_;
  std::vector<std::string> packed_data_names_;
  std::shared_ptr<XNNWorkspace> workspace_;
  // Owned so the cache outlives delete_packed_data in destroy(),
  // even when every other executor sharing it is gone. Empty when no
  // file-backed cache is in use.
  std::shared_ptr<XNNWeightsCache> weights_cache_;
  std::atomic<bool> in_use_{false};
  std::atomic<bool> destroyed_{false};

 public:
  XNNExecutor(std::shared_ptr<XNNWorkspace> workspace)
      : workspace_(workspace) {}

  ~XNNExecutor() {
    ET_DCHECK_MSG(
        !in_use_.load(std::memory_order_acquire),
        "XNNExecutor destroyed while in use");
    destroyed_.store(true, std::memory_order_release);
  }

  inline size_t getNumInputs() {
    return input_ids_.size();
  }

  inline size_t getNumOutputs() {
    return output_ids_.size();
  }

  inline std::vector<std::string> get_packed_data_names() {
    return packed_data_names_;
  }

  inline bool uses_weight_cache() const {
    return !packed_data_names_.empty();
  }

  inline std::shared_ptr<XNNWorkspace> get_workspace() {
    return workspace_;
  }

  // Set once by XNNPACKBackend::init after compileModel succeeds. Pass
  // an empty shared_ptr if no file-backed cache is in use for this PTE
  // (treated identically to never calling this).
  inline void set_weights_cache(std::shared_ptr<XNNWeightsCache> cache) {
    weights_cache_ = std::move(cache);
  }

  // Returns the per-PTE weights cache shared_ptr (may be empty). Used
  // by XNNPACKBackend::execute to lock the cache's mutex around runtime
  // invocation, and by destroy() to invoke delete_packed_data.
  inline std::shared_ptr<XNNWeightsCache> get_weights_cache() const {
    return weights_cache_;
  }

  /**
   * Initialize the XNNExecutor with a given runtime and input/output ids.
   * The input/output ids are expected to be sorted in order of their
   * flatbuffer id_outs
   */
  ET_NODISCARD executorch::runtime::Error initialize(
      xnn_runtime_t runtime,
      std::vector<uint32_t>&& input_ids,
      std::vector<uint32_t>&& output_ids,
      std::vector<std::string>&& packed_data_names);

  /**
   * Prepares the arguments for runtime graph execution.
   * args is an array of EValues that will be passed into the runtime.
   * input shapes will be propagated through the runtime, and perform
   * any additional memory planning as needed
   */
  ET_NODISCARD executorch::runtime::Error prepare_args(
      executorch::runtime::Span<executorch::runtime::EValue*> args);

  /**
   * Executes the graph using the args prepared at prepare_args().
   */
  ET_NODISCARD executorch::runtime::Error forward(
      executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext& context);

  /**
   * Resizes output tensors to match XNNPACK's computed shapes.
   *
   */
  ET_NODISCARD executorch::runtime::Error resize_outputs(
      executorch::runtime::Span<executorch::runtime::EValue*> args) const;

  /**
   * Converts output data types after XNNPACK execution.
   *
   * For arg_max pooling, XNNPACK outputs int32 index tensors that need
   * to be converted to int64 for ExecuTorch.
   */
  ET_NODISCARD executorch::runtime::Error convert_outputs(
      executorch::runtime::Span<executorch::runtime::EValue*> args) const;

  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
