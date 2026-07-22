/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

thread_local WebGPUExecutionOptions execution_options;

} // namespace

WebGPUExecutionOptions current_webgpu_execution_options() {
  return execution_options;
}

ScopedWebGPUExecutionOptions::ScopedWebGPUExecutionOptions(
    WebGPUExecutionOptions options)
    : previous_(execution_options) {
  execution_options = options;
}

ScopedWebGPUExecutionOptions::~ScopedWebGPUExecutionOptions() {
  execution_options = previous_;
}

WebGPUExecutionPlan plan_webgpu_execution(
    size_t dispatch_count,
    size_t output_count,
    ExecuteConfig config,
    const std::vector<SuppressibleOutput>& suppressible_outputs,
    WebGPUGraphExecutionOptions options) {
  std::vector<bool> suppressed_dispatches(dispatch_count, false);
  std::vector<bool> copy_outputs(output_count, true);
  std::vector<bool> suppressed_outputs(output_count, false);

  for (const auto& output : suppressible_outputs) {
    if (output.output_ordinal >= output_count ||
        output.dispatch_begin >= output.dispatch_end ||
        output.dispatch_end > dispatch_count) {
      throw std::runtime_error(
          "WebGPU: invalid suppressible output range (output_id " +
          std::to_string(output.output_id) + ")");
    }
    if (suppressed_outputs[output.output_ordinal]) {
      throw std::runtime_error(
          "WebGPU: duplicate suppressible output (output_id " +
          std::to_string(output.output_id) + ")");
    }
    suppressed_outputs[output.output_ordinal] = true;
    if (output.output_ordinal != options.suppress_output_ordinal) {
      continue;
    }
    copy_outputs[output.output_ordinal] = false;
    // Only the one ordinal matching suppress_output_ordinal reaches here (the
    // duplicate check above rejects a repeat), so its dispatch range is
    // disjoint by construction — mark it suppressed without a redundant overlap
    // check.
    for (size_t i = output.dispatch_begin; i < output.dispatch_end; i++) {
      suppressed_dispatches[i] = true;
    }
  }

  WebGPUExecutionPlan plan;
  plan.copy_outputs = std::move(copy_outputs);

  auto append_chunk = [&](size_t begin, size_t end) {
    std::vector<size_t> indices;
    indices.reserve(end - begin);
    for (size_t i = begin; i < end; i++) {
      if (!suppressed_dispatches[i]) {
        indices.push_back(i);
      }
    }
    if (!indices.empty()) {
      plan.dispatch_chunks.push_back(std::move(indices));
    }
  };

  if (config.chunk_size == 0 || dispatch_count <= config.chunk_size) {
    append_chunk(0, dispatch_count);
  } else {
    size_t start = 0;
    size_t current_chunk = config.initial_chunk_size > 0
        ? config.initial_chunk_size
        : config.chunk_size;
    while (start < dispatch_count) {
      const size_t end = std::min(start + current_chunk, dispatch_count);
      append_chunk(start, end);
      start = end;
      current_chunk = config.chunk_size;
    }
  }
  if (plan.dispatch_chunks.empty()) {
    plan.dispatch_chunks.emplace_back();
  }
  return plan;
}

WebGPUGraphExecutionOptions resolve_webgpu_graph_execution_options(
    const std::vector<const void*>& delegate_outputs,
    WebGPUExecutionOptions options) {
  if (options.discardable_output_data == nullptr) {
    return {};
  }
  if (!options.exact_method_certificate_verified) {
    return {};
  }

  size_t match = kNoOutputOrdinal;
  for (size_t i = 0; i < delegate_outputs.size(); i++) {
    if (delegate_outputs[i] != options.discardable_output_data) {
      continue;
    }
    if (match != kNoOutputOrdinal) {
      return {};
    }
    match = i;
  }
  return {match};
}

} // namespace executorch::backends::webgpu
