/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace executorch::backends::webgpu {

constexpr size_t kNoOutputOrdinal = static_cast<size_t>(-1);

struct WebGPUExecutionOptions {
  // The certificate must bind the exact PTE and method and prove one delegate,
  // no portable nodes, and a unique leaf method output at this data pointer.
  const void* discardable_output_data = nullptr;
  bool exact_method_certificate_verified = false;
};

struct WebGPUGraphExecutionOptions {
  size_t suppress_output_ordinal = kNoOutputOrdinal;
};

struct ExecuteConfig {
  size_t chunk_size = 0;
  size_t initial_chunk_size = 0;
};

struct SuppressibleOutput {
  int output_id = -1;
  size_t output_ordinal = 0;
  size_t dispatch_begin = 0;
  size_t dispatch_end = 0;
};

struct WebGPUExecutionPlan {
  std::vector<std::vector<size_t>> dispatch_chunks;
  std::vector<bool> copy_outputs;
};

WebGPUExecutionPlan plan_webgpu_execution(
    size_t dispatch_count,
    size_t output_count,
    ExecuteConfig config,
    const std::vector<SuppressibleOutput>& suppressible_outputs,
    WebGPUGraphExecutionOptions options,
    const std::vector<bool>& enabled_dispatches = {});

WebGPUGraphExecutionOptions resolve_webgpu_graph_execution_options(
    const std::vector<const void*>& delegate_outputs,
    WebGPUExecutionOptions options);

WebGPUExecutionOptions current_webgpu_execution_options();

class ScopedWebGPUExecutionOptions final {
 public:
  explicit ScopedWebGPUExecutionOptions(WebGPUExecutionOptions options);
  ~ScopedWebGPUExecutionOptions();

  ScopedWebGPUExecutionOptions(const ScopedWebGPUExecutionOptions&) = delete;
  ScopedWebGPUExecutionOptions& operator=(const ScopedWebGPUExecutionOptions&) =
      delete;
  ScopedWebGPUExecutionOptions(ScopedWebGPUExecutionOptions&&) = delete;
  ScopedWebGPUExecutionOptions& operator=(ScopedWebGPUExecutionOptions&&) =
      delete;

 private:
  WebGPUExecutionOptions previous_;
};

template <typename Fn>
decltype(auto) with_webgpu_execution_options(
    WebGPUExecutionOptions options,
    Fn&& fn) {
  ScopedWebGPUExecutionOptions scope(options);
  return std::forward<Fn>(fn)();
}

} // namespace executorch::backends::webgpu
