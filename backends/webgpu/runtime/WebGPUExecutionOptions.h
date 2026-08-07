/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace executorch::backends::webgpu {

constexpr size_t kNoOutputOrdinal = static_cast<size_t>(-1);

struct WebGPUExecutionOptions {
  // The certificate must bind the exact PTE and method and prove one delegate,
  // no portable nodes, and a unique leaf method output at this data pointer.
  // The caller must keep this pointer valid and unchanged for the complete
  // synchronous backend invocation in which these options are scoped.
  const void* discardable_output_data = nullptr;
  bool exact_method_certificate_verified = false;
  bool single_compute_pass = false;
  size_t max_compute_dispatches_per_pass = 0;
};

struct WebGPUGraphExecutionOptions {
  size_t suppress_output_ordinal = kNoOutputOrdinal;
  bool single_compute_pass = false;
  size_t max_compute_dispatches_per_pass = 0;
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
  bool single_compute_pass = false;
  size_t max_compute_dispatches_per_pass = 0;
};

enum class WebGPUCommandKind { Compute, GraphCopy, OutputCopy };

struct WebGPUCommandRecord {
  WebGPUCommandKind kind = WebGPUCommandKind::Compute;
  size_t static_dispatch_index = 0;
  size_t output_ordinal = kNoOutputOrdinal;
  std::string identity;
  bool enabled = true;
  bool zero_grid = false;
  bool suppressed = false;
  uint32_t workgroup_count_x = 1;
  uint32_t workgroup_count_y = 1;
  int64_t source_identity = -1;
  int64_t destination_identity = -1;
  size_t source_offset = 0;
  size_t destination_offset = 0;
  size_t byte_count = 0;
};

struct WebGPUCommandInventory {
  size_t static_dispatch_records = 0;
  size_t active_compute_count = 0;
  size_t zero_grid_compute_count = 0;
  size_t graph_copy_count = 0;
  size_t output_copy_count = 0;
  size_t maximal_compute_runs = 0;
  std::string canonical_commands_json;
};

struct WebGPUExecutionAttestation {
  uint64_t execution_ordinal = 0;
  bool requested = false;
  bool applied = false;
  bool completed = false;
  size_t encoded_compute_passes = 0;
  size_t queue_submit_count = 0;
  size_t max_compute_dispatches_per_pass = 0;
  std::string error_reason;
  WebGPUCommandInventory inventory;
};

bool webgpu_pass_cap_reached(
    size_t dispatches_in_current_pass,
    size_t max_compute_dispatches_per_pass);

size_t count_webgpu_compute_passes(
    const std::vector<WebGPUCommandRecord>& commands,
    bool single_compute_pass,
    size_t max_compute_dispatches_per_pass);

WebGPUCommandInventory build_webgpu_command_inventory(
    const std::vector<WebGPUCommandRecord>& commands);

std::string serialize_webgpu_execution_attestation(
    const WebGPUExecutionAttestation& attestation);

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
