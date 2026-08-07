/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

thread_local WebGPUExecutionOptions execution_options;

void append_json_string(std::ostringstream& out, const std::string& value) {
  out << '"';
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          constexpr char kHex[] = "0123456789abcdef";
          out << "\\u00" << kHex[c >> 4] << kHex[c & 0x0f];
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  out << '"';
}

const char* json_bool(bool value) {
  return value ? "true" : "false";
}

} // namespace

WebGPUCommandInventory build_webgpu_command_inventory(
    const std::vector<WebGPUCommandRecord>& commands) {
  WebGPUCommandInventory inventory;
  std::vector<int64_t> run_ordinals(commands.size(), -1);
  bool inside_compute_run = false;
  for (size_t i = 0; i < commands.size(); i++) {
    const auto& command = commands[i];
    if (command.kind != WebGPUCommandKind::OutputCopy) {
      ++inventory.static_dispatch_records;
    }
    if (command.kind == WebGPUCommandKind::Compute) {
      if (command.zero_grid) {
        ++inventory.zero_grid_compute_count;
      }
      if (!command.enabled || command.zero_grid) {
        continue;
      }
      if (!inside_compute_run) {
        ++inventory.maximal_compute_runs;
        inside_compute_run = true;
      }
      run_ordinals[i] =
          static_cast<int64_t>(inventory.maximal_compute_runs - 1);
      ++inventory.active_compute_count;
      continue;
    }
    if (command.kind == WebGPUCommandKind::GraphCopy) {
      if (command.enabled) {
        ++inventory.graph_copy_count;
        inside_compute_run = false;
      }
      continue;
    }
    if (command.enabled && !command.suppressed) {
      ++inventory.output_copy_count;
    }
    inside_compute_run = false;
  }

  std::vector<int64_t> preceding_runs(commands.size(), -1);
  std::vector<int64_t> following_runs(commands.size(), -1);
  int64_t last_run = -1;
  for (size_t i = 0; i < commands.size(); i++) {
    preceding_runs[i] = last_run;
    if (run_ordinals[i] >= 0) {
      last_run = run_ordinals[i];
    }
  }
  int64_t next_run = -1;
  for (size_t i = commands.size(); i > 0; i--) {
    following_runs[i - 1] = next_run;
    if (run_ordinals[i - 1] >= 0) {
      next_run = run_ordinals[i - 1];
    }
  }

  std::ostringstream out;
  out << "{\"commands\":[";
  for (size_t i = 0; i < commands.size(); i++) {
    if (i != 0) {
      out << ',';
    }
    const auto& command = commands[i];
    if (command.kind == WebGPUCommandKind::Compute) {
      out << "{\"enabled\":" << json_bool(command.enabled)
          << ",\"expectedMaximalRunOrdinal\":" << run_ordinals[i]
          << ",\"grid\":[" << command.workgroup_count_x << ','
          << command.workgroup_count_y << ",1],\"identity\":";
      append_json_string(out, command.identity);
      out << ",\"kind\":\"compute\",\"staticDispatchIndex\":"
          << command.static_dispatch_index << ",\"zeroGrid\":"
          << json_bool(command.zero_grid) << '}';
    } else if (command.kind == WebGPUCommandKind::GraphCopy) {
      out << "{\"byteCount\":" << command.byte_count
          << ",\"destinationIdentity\":" << command.destination_identity
          << ",\"destinationOffset\":" << command.destination_offset
          << ",\"enabled\":" << json_bool(command.enabled)
          << ",\"followingMaximalRunOrdinal\":" << following_runs[i]
          << ",\"kind\":\"graph_copy\",\"precedingMaximalRunOrdinal\":"
          << preceding_runs[i] << ",\"sourceIdentity\":"
          << command.source_identity << ",\"sourceOffset\":"
          << command.source_offset << ",\"staticDispatchIndex\":"
          << command.static_dispatch_index << '}';
    } else {
      out << "{\"byteCount\":" << command.byte_count
          << ",\"destinationIdentity\":" << command.destination_identity
          << ",\"destinationOffset\":" << command.destination_offset
          << ",\"enabled\":" << json_bool(command.enabled)
          << ",\"kind\":\"output_copy\",\"outputOrdinal\":"
          << command.output_ordinal << ",\"sourceIdentity\":"
          << command.source_identity << ",\"sourceOffset\":"
          << command.source_offset << ",\"suppressed\":"
          << json_bool(command.suppressed) << '}';
    }
  }
  out << "],\"schemaVersion\":1}";
  inventory.canonical_commands_json = out.str();
  return inventory;
}

std::string serialize_webgpu_execution_attestation(
    const WebGPUExecutionAttestation& attestation) {
  const auto& inventory = attestation.inventory;
  std::ostringstream out;
  out << "{\"activeComputeCount\":" << inventory.active_compute_count
      << ",\"applied\":" << json_bool(attestation.applied)
      << ",\"canonicalCommands\":"
      << (inventory.canonical_commands_json.empty()
              ? "{\"commands\":[],\"schemaVersion\":1}"
              : inventory.canonical_commands_json)
      << ",\"completed\":" << json_bool(attestation.completed)
      << ",\"encodedComputePasses\":"
      << attestation.encoded_compute_passes << ",\"errorReason\":";
  append_json_string(out, attestation.error_reason);
  out << ",\"executionOrdinal\":" << attestation.execution_ordinal
      << ",\"graphCopyCount\":" << inventory.graph_copy_count
      << ",\"maxComputeDispatchesPerPass\":"
      << attestation.max_compute_dispatches_per_pass
      << ",\"maximalComputeRuns\":" << inventory.maximal_compute_runs
      << ",\"outputCopyCount\":" << inventory.output_copy_count
      << ",\"queueSubmitCount\":" << attestation.queue_submit_count
      << ",\"requested\":" << json_bool(attestation.requested)
      << ",\"schemaVersion\":1,\"staticDispatchRecords\":"
      << inventory.static_dispatch_records << ",\"zeroGridComputeCount\":"
      << inventory.zero_grid_compute_count << '}';
  return out.str();
}

bool webgpu_pass_cap_reached(
    size_t dispatches_in_current_pass,
    size_t max_compute_dispatches_per_pass) {
  return max_compute_dispatches_per_pass != 0 &&
      dispatches_in_current_pass >= max_compute_dispatches_per_pass;
}

size_t count_webgpu_compute_passes(
    const std::vector<WebGPUCommandRecord>& commands,
    bool single_compute_pass,
    size_t max_compute_dispatches_per_pass) {
  if (!single_compute_pass && max_compute_dispatches_per_pass != 0) {
    throw std::invalid_argument(
        "WebGPU: pass cap requires single_compute_pass");
  }
  size_t pass_count = 0;
  size_t dispatches_in_pass = 0;
  for (const auto& command : commands) {
    if (command.kind != WebGPUCommandKind::Compute) {
      if (command.enabled) {
        dispatches_in_pass = 0;
      }
      continue;
    }
    if (!command.enabled || command.zero_grid) {
      continue;
    }
    if (!single_compute_pass || dispatches_in_pass == 0) {
      ++pass_count;
    }
    ++dispatches_in_pass;
    if (!single_compute_pass || webgpu_pass_cap_reached(
                                    dispatches_in_pass,
                                    max_compute_dispatches_per_pass)) {
      dispatches_in_pass = 0;
    }
  }
  return pass_count;
}

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
    WebGPUGraphExecutionOptions options,
    const std::vector<bool>& enabled_dispatches) {
  if (!enabled_dispatches.empty() &&
      enabled_dispatches.size() != dispatch_count) {
    throw std::runtime_error("WebGPU: enabled dispatch count mismatch");
  }
  std::vector<bool> suppressed_dispatches(dispatch_count, false);
  std::vector<bool> copy_outputs(output_count, true);
  std::vector<bool> seen_output_ordinals(output_count, false);

  for (const auto& output : suppressible_outputs) {
    if (output.output_ordinal >= output_count ||
        output.dispatch_begin >= output.dispatch_end ||
        output.dispatch_end > dispatch_count) {
      throw std::runtime_error(
          "WebGPU: invalid suppressible output range (output_id " +
          std::to_string(output.output_id) + ")");
    }
    if (seen_output_ordinals[output.output_ordinal]) {
      throw std::runtime_error(
          "WebGPU: duplicate suppressible output (output_id " +
          std::to_string(output.output_id) + ")");
    }
    seen_output_ordinals[output.output_ordinal] = true;
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
  plan.single_compute_pass = options.single_compute_pass;
  plan.max_compute_dispatches_per_pass =
      options.max_compute_dispatches_per_pass;

  auto append_chunk = [&](size_t begin, size_t end) {
    std::vector<size_t> indices;
    indices.reserve(end - begin);
    for (size_t i = begin; i < end; i++) {
      if (!suppressed_dispatches[i] &&
          (enabled_dispatches.empty() || enabled_dispatches[i])) {
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
  if (plan.dispatch_chunks.empty() &&
      std::any_of(
          plan.copy_outputs.begin(), plan.copy_outputs.end(), [](bool copy) {
            return copy;
          })) {
    plan.dispatch_chunks.emplace_back();
  }
  return plan;
}

WebGPUGraphExecutionOptions resolve_webgpu_graph_execution_options(
    const std::vector<const void*>& delegate_outputs,
    WebGPUExecutionOptions options) {
  WebGPUGraphExecutionOptions resolved;
  resolved.single_compute_pass = options.single_compute_pass;
  resolved.max_compute_dispatches_per_pass =
      options.max_compute_dispatches_per_pass;
  if (options.discardable_output_data == nullptr) {
    return resolved;
  }
  if (!options.exact_method_certificate_verified) {
    return resolved;
  }

  size_t match = kNoOutputOrdinal;
  for (size_t i = 0; i < delegate_outputs.size(); i++) {
    if (delegate_outputs[i] != options.discardable_output_data) {
      continue;
    }
    if (match != kNoOutputOrdinal) {
      return resolved;
    }
    match = i;
  }
  resolved.suppress_output_ordinal = match;
  return resolved;
}

} // namespace executorch::backends::webgpu
