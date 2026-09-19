/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/passes/SwiGLU.h>

#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu::passes {

namespace {

constexpr const char* kQ4gswLinearOpName = "et_vk.linear_q4gsw.default";
constexpr const char* kSigmoidOpName = "aten.sigmoid.default";
constexpr const char* kMulOpName = "aten.mul.Tensor";

// Uniform layout matching silu_mul_fused.wgsl's Params struct.
struct SiluMulParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

uint32_t checked_silu_mul_numel(const std::vector<int64_t>& dims) {
  const uint64_t numel = utils::numel_of(dims);
  if (numel == 0 || numel > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("silu_mul_fused: element count out of range");
  }
  return static_cast<uint32_t>(numel);
}

void resize_silu_mul_fused(
    WebGPUGraph& graph,
    int gate_id,
    int up_id,
    int out_id,
    uint32_t wg_size,
    size_t dispatch_idx,
    WGPUBuffer params_buffer) {
  const auto& gate_dims = graph.cur_dims(gate_id);
  const auto& up_dims = graph.cur_dims(up_id);
  if (gate_dims != up_dims) {
    throw std::runtime_error("silu_mul_fused(resize): gate/up shape mismatch");
  }
  const uint32_t live_numel = checked_silu_mul_numel(gate_dims);
  const size_t live_nbytes = static_cast<size_t>(live_numel) * sizeof(float);
  if (graph.get_tensor(gate_id).cur_nbytes != live_nbytes ||
      graph.get_tensor(up_id).cur_nbytes != live_nbytes) {
    throw std::runtime_error(
        "silu_mul_fused(resize): gate/up byte-size mismatch");
  }
  graph.set_cur_dims(out_id, gate_dims);
  const SiluMulParams params = {live_numel, {0u, 0u, 0u}};
  wgpuQueueWriteBuffer(
      graph.queue(), params_buffer, 0, &params, sizeof(params));
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      graph.device(), live_numel, wg_size, "silu_mul_fused(resize)");
  auto& dispatch = graph.dispatch_at(dispatch_idx);
  dispatch.workgroup_count_x = workgroup_count.x;
  dispatch.workgroup_count_y = workgroup_count.y;
}

} // namespace

void detect_swiglu_fusions(
    const WebGPUGraph& graph,
    const vkgraph::VkGraph* fb_graph,
    int num_vals,
    std::vector<SwiGluFusion>& fusions,
    std::unordered_map<unsigned, size_t>& gate_producers,
    std::unordered_map<unsigned, size_t>& anchors,
    std::unordered_set<unsigned>& skipped_ops,
    std::unordered_set<unsigned>& claimed_ops) {
  const auto* chain = fb_graph->chain();
  // Detect only the exact q4 gate/up MLP pattern. Full-chain occurrence and
  // definition counts make every folded intermediate private to the pattern.
  if (!chain) {
    return;
  }
  struct ExactUnary {
    unsigned op;
    int input;
  };
  struct ExactBinary {
    unsigned op;
    int lhs;
    int rhs;
  };
  std::vector<size_t> occurrences(num_vals, 0);
  std::vector<size_t> definitions(num_vals, 0);
  std::vector<int> producer(num_vals, -1);
  std::unordered_map<int, unsigned> q4_by_output;
  std::unordered_map<int, ExactUnary> sigmoid_by_output;
  std::unordered_map<int, ExactBinary> mul_by_output;

  auto count_occurrence = [&](int id) {
    if (id < 0 || id >= num_vals) {
      return;
    }
    occurrences[id]++;
    if (graph.get_value_type(id) == WebGPUGraph::ValueType::ValueList) {
      for (int member : graph.get_value_list(id)) {
        if (member >= 0 && member < num_vals) {
          occurrences[member]++;
        }
      }
    }
  };

  for (unsigned i = 0; i < chain->size(); i++) {
    const auto* op = chain->Get(i);
    const auto* args = op->args();
    if (!args || args->size() == 0) {
      continue;
    }
    for (unsigned j = 0; j < args->size(); j++) {
      count_occurrence(static_cast<int>(args->Get(j)));
    }
    const std::string name = op->name()->str();
    int output = -1;
    if (name == kQ4gswLinearOpName && args->size() == 6) {
      output = static_cast<int>(args->Get(5));
      q4_by_output[output] = i;
    } else if (name == kSigmoidOpName && args->size() == 2) {
      output = static_cast<int>(args->Get(1));
      sigmoid_by_output[output] = {i, static_cast<int>(args->Get(0))};
    } else if (name == kMulOpName && args->size() == 3) {
      output = static_cast<int>(args->Get(2));
      mul_by_output[output] = {
          i, static_cast<int>(args->Get(0)), static_cast<int>(args->Get(1))};
    }
    if (output >= 0 && output < num_vals) {
      definitions[output]++;
      producer[output] = static_cast<int>(i);
    }
  }

  const auto& output_ids = graph.output_ids();
  auto is_graph_output = [&](int id) {
    return std::find(output_ids.begin(), output_ids.end(), id) !=
        output_ids.end();
  };
  for (unsigned mul2_op = 0; mul2_op < chain->size(); mul2_op++) {
    const auto* mul2_call = chain->Get(mul2_op);
    const auto* mul2_args = mul2_call->args();
    if (mul2_call->name()->str() != kMulOpName || !mul2_args ||
        mul2_args->size() != 3) {
      continue;
    }

    std::vector<SwiGluFusion> candidates;
    auto try_orientation = [&](int silu_id, int up_id) {
      const auto mul1_it = mul_by_output.find(silu_id);
      if (mul1_it == mul_by_output.end()) {
        return;
      }
      const ExactBinary& mul1 = mul1_it->second;
      int gate_id = -1;
      int sigmoid_id = -1;
      unsigned sigmoid_op = 0;
      const auto lhs_sig = sigmoid_by_output.find(mul1.lhs);
      const auto rhs_sig = sigmoid_by_output.find(mul1.rhs);
      if (lhs_sig != sigmoid_by_output.end() &&
          lhs_sig->second.input == mul1.rhs) {
        gate_id = mul1.rhs;
        sigmoid_id = mul1.lhs;
        sigmoid_op = lhs_sig->second.op;
      } else if (
          rhs_sig != sigmoid_by_output.end() &&
          rhs_sig->second.input == mul1.lhs) {
        gate_id = mul1.lhs;
        sigmoid_id = mul1.rhs;
        sigmoid_op = rhs_sig->second.op;
      } else {
        return;
      }

      const auto gate_q4 = q4_by_output.find(gate_id);
      const auto up_q4 = q4_by_output.find(up_id);
      if (gate_q4 == q4_by_output.end() || up_q4 == q4_by_output.end() ||
          gate_q4->second == up_q4->second) {
        return;
      }
      const auto* gate_args = chain->Get(gate_q4->second)->args();
      const auto* up_args = chain->Get(up_q4->second)->args();
      if (!gate_args || !up_args || gate_args->size() != 6 ||
          up_args->size() != 6 || gate_args->Get(0) != up_args->Get(0) ||
          static_cast<int>(gate_args->Get(5)) != gate_id ||
          static_cast<int>(up_args->Get(5)) != up_id) {
        return;
      }
      const int common_input_id = static_cast<int>(gate_args->Get(0));
      const int out_id = static_cast<int>(mul2_args->Get(2));
      const int ids[] = {gate_id, up_id, sigmoid_id, silu_id, out_id};
      std::unordered_set<int> distinct_ids(std::begin(ids), std::end(ids));
      if (distinct_ids.size() != 5 || common_input_id < 0 ||
          common_input_id >= num_vals) {
        return;
      }
      for (int id : ids) {
        if (id < 0 || id >= num_vals ||
            graph.get_value_type(id) != WebGPUGraph::ValueType::Tensor ||
            definitions[id] != 1) {
          return;
        }
      }
      if (producer[gate_id] != static_cast<int>(gate_q4->second) ||
          producer[up_id] != static_cast<int>(up_q4->second) ||
          producer[sigmoid_id] != static_cast<int>(sigmoid_op) ||
          producer[silu_id] != static_cast<int>(mul1.op) ||
          producer[out_id] != static_cast<int>(mul2_op) ||
          occurrences[gate_id] != 3 || occurrences[up_id] != 2 ||
          occurrences[sigmoid_id] != 2 || occurrences[silu_id] != 2) {
        return;
      }
      if (!(gate_q4->second < sigmoid_op && sigmoid_op < mul1.op &&
            mul1.op < mul2_op && up_q4->second < mul2_op)) {
        return;
      }
      if (is_graph_output(gate_id) || is_graph_output(sigmoid_id) ||
          is_graph_output(silu_id) || graph.mem_obj_id(gate_id) < 0) {
        return;
      }

      const auto& gate = graph.get_tensor(gate_id);
      const auto& up = graph.get_tensor(up_id);
      const auto& sigmoid = graph.get_tensor(sigmoid_id);
      const auto& silu = graph.get_tensor(silu_id);
      const auto& out = graph.get_tensor(out_id);
      if (!utils::is_fp32_tensor(gate) || !utils::is_fp32_tensor(up) ||
          !utils::is_fp32_tensor(sigmoid) || !utils::is_fp32_tensor(silu) ||
          !utils::is_fp32_tensor(out) || gate.dims != up.dims ||
          gate.dims != sigmoid.dims || gate.dims != silu.dims ||
          gate.dims != out.dims || gate.nbytes != up.nbytes ||
          gate.nbytes != sigmoid.nbytes || gate.nbytes != silu.nbytes ||
          gate.nbytes != out.nbytes || up.buffer == out.buffer ||
          gate.buffer == out.buffer) {
        return;
      }
      candidates.push_back(
          {common_input_id,
           gate_id,
           up_id,
           sigmoid_id,
           silu_id,
           out_id,
           gate_q4->second,
           sigmoid_op,
           mul1.op,
           mul2_op});
    };

    try_orientation(
        static_cast<int>(mul2_args->Get(0)),
        static_cast<int>(mul2_args->Get(1)));
    try_orientation(
        static_cast<int>(mul2_args->Get(1)),
        static_cast<int>(mul2_args->Get(0)));
    if (candidates.size() != 1) {
      continue;
    }
    const SwiGluFusion& fusion = candidates.front();
    const unsigned pattern_ops[] = {
        fusion.gate_op,
        q4_by_output.at(fusion.up_id),
        fusion.sigmoid_op,
        fusion.mul1_op,
        fusion.mul2_op};
    bool overlaps = false;
    for (unsigned op : pattern_ops) {
      overlaps = overlaps || claimed_ops.count(op) != 0;
    }
    if (overlaps) {
      continue;
    }
    const size_t fusion_idx = fusions.size();
    fusions.push_back(fusion);
    gate_producers[fusion.gate_op] = fusion_idx;
    anchors[fusion.mul2_op] = fusion_idx;
    skipped_ops.insert(fusion.sigmoid_op);
    skipped_ops.insert(fusion.mul1_op);
    // mul2_op is the fusion anchor: its Phase-3 branch emits the fused
    // dispatch and continues before the skipped-ops check, so it needs no
    // skipped_ops entry.
    for (unsigned op : pattern_ops) {
      claimed_ops.insert(op);
    }
  }
}

void add_silu_mul_fused_dispatch(
    WebGPUGraph& graph,
    int common_input_id,
    int gate_id,
    int up_id,
    int out_id) {
  const auto& gate = graph.get_tensor(gate_id);
  const auto& up = graph.get_tensor(up_id);
  const auto& out = graph.get_tensor(out_id);
  const uint32_t num_elements = checked_silu_mul_numel(gate.dims);
  const uint32_t wg_size = utils::clamp_workgroup_size(
      graph.device(),
      get_webgpu_shader_info("silu_mul_fused").workgroup_size_x);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      graph.device(), num_elements, wg_size, "silu_mul_fused");

  const SiluMulParams params = {num_elements, {0u, 0u, 0u}};
  WGPUBuffer params_buffer = graph.create_params_buffer(params);
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = "silu_mul_fused";
  descriptor.bindings = {
      {gate.buffer, 0u, gate.nbytes},
      {up.buffer, 0u, up.nbytes},
      {out.buffer, 0u, out.nbytes},
      {params_buffer, 0u, sizeof(SiluMulParams)}};
  descriptor.constants = {{"wg_size", static_cast<double>(wg_size)}};
  descriptor.grid = {workgroup_count.x, workgroup_count.y};
  const size_t dispatch_idx = graph.add_compute_dispatch(descriptor);

  graph.add_tensor_resize_hook(
      common_input_id,
      [gate_id, up_id, out_id, wg_size, dispatch_idx, params_buffer](
          WebGPUGraph& g) {
        resize_silu_mul_fused(
            g, gate_id, up_id, out_id, wg_size, dispatch_idx, params_buffer);
      });
}

} // namespace executorch::backends::webgpu::passes
