/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/passes/QkvBk64.h>

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace executorch::backends::webgpu::passes {

namespace {

constexpr const char* kQ4gswLinearOpName = "et_vk.linear_q4gsw.default";

constexpr uint32_t kQkvQWidth = 2048u;
constexpr uint32_t kQkvKvWidth = 512u;
constexpr uint32_t kQkvFusedWidth = 3072u;
constexpr uint32_t kQkvK = 2048u;
constexpr uint32_t kQkvKPacked = 1024u;
constexpr uint32_t kQkvGroupSize = 64u;
constexpr uint32_t kQkvNumGroups = 32u;
constexpr uint32_t kQkvTile = 64u;

// Uniform layout matching q4gsw_qkv_bk64.wgsl's Params struct.
struct QkvBk64Params {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(QkvBk64Params) == 32);

bool is_qkv_bk64_live_m(uint32_t m) {
  return m == 128u || m == 508u || m == 512u;
}

struct QkvBk64ResizeContext {
  int input_id;
  std::array<int, 3> output_ids;
  std::array<size_t, 3> separate_begin;
  std::array<size_t, 3> separate_end;
  size_t fused_dispatch;
  uint32_t max_m;
  WGPUBuffer params_buffer;
};

void resize_qkv_bk64(WebGPUGraph& graph, const QkvBk64ResizeContext& context) {
  const auto& input_dims = graph.cur_dims(context.input_id);
  const uint64_t input_numel = utils::numel_of(input_dims);
  if (input_dims.empty() || input_numel % kQkvK != 0u) {
    throw std::runtime_error(
        "linear_q4gsw_bk64_qkv(resize): malformed input shape");
  }
  const uint64_t live_m = input_numel / kQkvK;
  if (live_m == 0u || live_m > context.max_m) {
    throw std::runtime_error(
        "linear_q4gsw_bk64_qkv(resize): live M out of range");
  }
  const uint32_t m = static_cast<uint32_t>(live_m);
  const uint32_t widths[3] = {kQkvQWidth, kQkvKvWidth, kQkvKvWidth};
  for (size_t i = 0; i < context.output_ids.size(); i++) {
    std::vector<int64_t> output_dims = input_dims;
    output_dims.back() = widths[i];
    graph.set_cur_dims(context.output_ids[i], output_dims);
  }

  const QkvBk64Params params = {
      m,
      kQkvFusedWidth,
      kQkvK,
      kQkvKPacked,
      kQkvGroupSize,
      kQkvFusedWidth,
      0u,
      0u};
  wgpuQueueWriteBuffer(
      graph.queue(), context.params_buffer, 0, &params, sizeof(params));

  const bool use_fused = is_qkv_bk64_live_m(m);
  auto& fused = graph.dispatch_at(context.fused_dispatch);
  fused.workgroup_count_x = use_fused
      ? ((m + kQkvTile - 1u) / kQkvTile) * (kQkvFusedWidth / kQkvTile)
      : 0u;
  fused.workgroup_count_y = use_fused ? 1u : 0u;
  if (use_fused) {
    // The separate projections are inactive while the fused route is live.
    for (size_t member = 0; member < context.separate_begin.size(); member++) {
      for (size_t i = context.separate_begin[member];
           i < context.separate_end[member];
           i++) {
        auto& dispatch = graph.dispatch_at(i);
        dispatch.workgroup_count_x = 0u;
        dispatch.workgroup_count_y = 0u;
      }
    }
  } else {
    // The separate projections' own resize hooks are registered before this
    // one (Phase 3 processes each Q/K/V member before this fusion's combined
    // hook) and unconditionally restore their live grids, so this hook
    // normally has nothing to do here. That ordering isn't enforced by the
    // type system, so fail loud rather than silently drop Q/K/V outputs if a
    // future change ever violates it.
    for (size_t member = 0; member < context.separate_begin.size(); member++) {
      for (size_t i = context.separate_begin[member];
           i < context.separate_end[member];
           i++) {
        const auto& dispatch = graph.dispatch_at(i);
        if (dispatch.workgroup_count_x == 0u ||
            dispatch.workgroup_count_y == 0u) {
          throw std::runtime_error(
              "linear_q4gsw_bk64_qkv(resize): separate projection dispatch "
              "was not restored before the QKV resize hook ran");
        }
      }
    }
  }
}

} // namespace

bool qkv_bk64_device_supported(WGPUDevice device) {
  WGPULimits limits = {};
  const WebGPUContext* context = get_default_webgpu_context();
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup >= 256u &&
      limits.maxComputeWorkgroupSizeX >= 16u &&
      limits.maxComputeWorkgroupSizeY >= 16u &&
      limits.maxComputeWorkgroupStorageSize >= 16384u &&
      limits.maxComputeWorkgroupsPerDimension >= 384u;
}

void detect_qkv_bk64_fusions(
    const WebGPUGraph& graph,
    const vkgraph::VkGraph* fb_graph,
    int num_vals,
    std::vector<QkvBk64Fusion>& fusions,
    std::unordered_map<unsigned, size_t>& first_ops,
    std::unordered_map<unsigned, size_t>& last_ops,
    std::unordered_map<unsigned, size_t>& member_ops) {
  const auto* chain = fb_graph->chain();
  if (!chain || !qkv_bk64_device_supported(graph.device())) {
    return;
  }
  std::unordered_map<int, std::vector<unsigned>> q4_ops_by_input;
  std::vector<int> input_order;
  for (unsigned i = 0; i < chain->size(); i++) {
    const auto* op = chain->Get(i);
    const auto* args = op->args();
    if (op->name()->str() != kQ4gswLinearOpName || !args || args->size() != 6) {
      continue;
    }
    const int input_id = static_cast<int>(args->Get(0));
    if (q4_ops_by_input.count(input_id) == 0) {
      input_order.push_back(input_id);
    }
    q4_ops_by_input[input_id].push_back(i);
  }

  auto op_arg = [&](unsigned op_index, unsigned arg_index) {
    return static_cast<int>(chain->Get(op_index)->args()->Get(arg_index));
  };
  const auto& output_ids = graph.output_ids();
  auto is_graph_output = [&](int id) {
    return std::find(output_ids.begin(), output_ids.end(), id) !=
        output_ids.end();
  };
  for (int input_id : input_order) {
    const auto& ops = q4_ops_by_input.at(input_id);
    if (ops.size() != 3 || input_id < 0 || input_id >= num_vals ||
        graph.get_value_type(input_id) != WebGPUGraph::ValueType::Tensor) {
      continue;
    }

    QkvBk64Fusion fusion;
    fusion.input_id = input_id;
    bool exact_args = true;
    for (size_t member = 0; member < 3; member++) {
      fusion.op_indices[member] = ops[member];
      fusion.weight_ids[member] = op_arg(ops[member], 1);
      fusion.scale_ids[member] = op_arg(ops[member], 2);
      fusion.output_ids[member] = op_arg(ops[member], 5);
      const int group_size_id = op_arg(ops[member], 3);
      const int bias_id = op_arg(ops[member], 4);
      exact_args = exact_args && group_size_id >= 0 &&
          group_size_id < num_vals &&
          graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int &&
          graph.get_int(group_size_id) == kQkvGroupSize && bias_id >= 0 &&
          bias_id < num_vals &&
          graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Null;
    }
    if (!exact_args) {
      continue;
    }

    const std::array<int, 6> constant_ids = {
        fusion.weight_ids[0],
        fusion.weight_ids[1],
        fusion.weight_ids[2],
        fusion.scale_ids[0],
        fusion.scale_ids[1],
        fusion.scale_ids[2]};
    const std::unordered_set<int> distinct_constants(
        constant_ids.begin(), constant_ids.end());
    bool direct_constants = distinct_constants.size() == constant_ids.size();
    for (int id : constant_ids) {
      direct_constants = direct_constants && id >= 0 && id < num_vals &&
          graph.get_value_type(id) == WebGPUGraph::ValueType::Tensor &&
          graph.has_constant_source(id) &&
          graph.get_tensor(id).buffer != nullptr;
    }
    if (!direct_constants) {
      continue;
    }

    const std::unordered_set<int> distinct_outputs = {
        fusion.output_ids[0], fusion.output_ids[1], fusion.output_ids[2]};
    bool outputs_ok = distinct_outputs.size() == 3;
    for (int id : fusion.output_ids) {
      outputs_ok = outputs_ok && id >= 0 && id < num_vals &&
          graph.get_value_type(id) == WebGPUGraph::ValueType::Tensor &&
          graph.mem_obj_id(id) >= 0 && !is_graph_output(id) &&
          utils::is_fp32_tensor(graph.get_tensor(id));
    }
    if (!outputs_ok) {
      continue;
    }

    const auto& input = graph.get_tensor(input_id);
    if (!utils::is_fp32_tensor(input) || input.dims.empty() ||
        input.dims.back() != kQkvK) {
      continue;
    }
    const uint64_t input_numel = utils::numel_of(input.dims);
    if (input_numel % kQkvK != 0u || input_numel / kQkvK < 128u ||
        input_numel / kQkvK > UINT32_MAX) {
      continue;
    }
    fusion.max_m = static_cast<uint32_t>(input_numel / kQkvK);

    const uint32_t widths[3] = {kQkvQWidth, kQkvKvWidth, kQkvKvWidth};
    bool exact_geometry = true;
    for (size_t member = 0; member < 3; member++) {
      const auto& weight = graph.get_tensor(fusion.weight_ids[member]);
      const auto& scale = graph.get_tensor(fusion.scale_ids[member]);
      const auto& output = graph.get_tensor(fusion.output_ids[member]);
      exact_geometry =
          exact_geometry && weight.dims.size() == 2 &&
          weight.dims[0] == widths[member] && weight.dims[1] == kQkvKPacked &&
          weight.nbytes == static_cast<size_t>(widths[member]) * kQkvKPacked &&
          scale.dims.size() == 2 && scale.dims[0] == kQkvNumGroups &&
          scale.dims[1] == widths[member] && utils::is_fp32_tensor(scale) &&
          output.dims.size() == input.dims.size() &&
          std::equal(
              input.dims.begin(), input.dims.end() - 1, output.dims.begin()) &&
          output.dims.back() == widths[member] &&
          utils::numel_of(output.dims) ==
              static_cast<uint64_t>(fusion.max_m) * widths[member];
    }
    if (!exact_geometry) {
      continue;
    }

    const size_t fusion_index = fusions.size();
    fusions.push_back(fusion);
    first_ops[ops[0]] = fusion_index;
    last_ops[ops[2]] = fusion_index;
    for (unsigned op : ops) {
      member_ops[op] = fusion_index;
    }
  }
}

void retain_unclaimed_qkv_fusions(
    std::vector<QkvBk64Fusion>& fusions,
    std::unordered_map<unsigned, size_t>& first_ops,
    std::unordered_map<unsigned, size_t>& last_ops,
    std::unordered_map<unsigned, size_t>& member_ops,
    std::unordered_set<unsigned>& claimed_ops) {
  std::vector<QkvBk64Fusion> retained_fusions;
  first_ops.clear();
  last_ops.clear();
  member_ops.clear();
  for (QkvBk64Fusion& fusion : fusions) {
    bool overlaps = false;
    for (unsigned op : fusion.op_indices) {
      overlaps = overlaps || claimed_ops.count(op) != 0;
    }
    if (overlaps) {
      continue;
    }
    const size_t fusion_index = retained_fusions.size();
    retained_fusions.push_back(std::move(fusion));
    const QkvBk64Fusion& retained = retained_fusions.back();
    first_ops[retained.op_indices[0]] = fusion_index;
    last_ops[retained.op_indices[2]] = fusion_index;
    for (unsigned op : retained.op_indices) {
      member_ops[op] = fusion_index;
      claimed_ops.insert(op);
    }
  }
  fusions = std::move(retained_fusions);
}

void add_qkv_bk64_dispatch(WebGPUGraph& graph, QkvBk64Fusion& fusion) {
  const auto& input = graph.get_tensor(fusion.input_id);
  const auto& output_q = graph.get_tensor(fusion.output_ids[0]);
  const auto& output_k = graph.get_tensor(fusion.output_ids[1]);
  const auto& output_v = graph.get_tensor(fusion.output_ids[2]);
  const auto& weight_q = graph.get_tensor(fusion.weight_ids[0]);
  const auto& weight_k = graph.get_tensor(fusion.weight_ids[1]);
  const auto& weight_v = graph.get_tensor(fusion.weight_ids[2]);
  const auto& scale_q = graph.get_tensor(fusion.scale_ids[0]);
  const auto& scale_k = graph.get_tensor(fusion.scale_ids[1]);
  const auto& scale_v = graph.get_tensor(fusion.scale_ids[2]);

  const size_t weight_row_bytes = kQkvKPacked;
  WGPUBuffer fused_weight = graph.create_scratch_buffer(
      static_cast<size_t>(kQkvFusedWidth) * weight_row_bytes);
  WGPUBuffer fused_scales = graph.create_scratch_buffer(
      static_cast<size_t>(kQkvNumGroups) * kQkvFusedWidth * sizeof(float));

  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(graph.device(), nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      weight_q.buffer,
      0,
      fused_weight,
      0,
      static_cast<uint64_t>(kQkvQWidth) * weight_row_bytes);
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      weight_k.buffer,
      0,
      fused_weight,
      static_cast<uint64_t>(kQkvQWidth) * weight_row_bytes,
      static_cast<uint64_t>(kQkvKvWidth) * weight_row_bytes);
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      weight_v.buffer,
      0,
      fused_weight,
      static_cast<uint64_t>(kQkvQWidth + kQkvKvWidth) * weight_row_bytes,
      static_cast<uint64_t>(kQkvKvWidth) * weight_row_bytes);
  for (uint32_t group = 0; group < kQkvNumGroups; group++) {
    const uint64_t destination =
        static_cast<uint64_t>(group) * kQkvFusedWidth * sizeof(float);
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        scale_q.buffer,
        static_cast<uint64_t>(group) * kQkvQWidth * sizeof(float),
        fused_scales,
        destination,
        static_cast<uint64_t>(kQkvQWidth) * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        scale_k.buffer,
        static_cast<uint64_t>(group) * kQkvKvWidth * sizeof(float),
        fused_scales,
        destination + static_cast<uint64_t>(kQkvQWidth) * sizeof(float),
        static_cast<uint64_t>(kQkvKvWidth) * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        scale_v.buffer,
        static_cast<uint64_t>(group) * kQkvKvWidth * sizeof(float),
        fused_scales,
        destination +
            static_cast<uint64_t>(kQkvQWidth + kQkvKvWidth) * sizeof(float),
        static_cast<uint64_t>(kQkvKvWidth) * sizeof(float));
  }
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, nullptr);
  wgpuQueueSubmit(graph.queue(), 1, &command);
  wgpuCommandBufferRelease(command);
  wgpuCommandEncoderRelease(encoder);

  const QkvBk64Params params = {
      fusion.max_m,
      kQkvFusedWidth,
      kQkvK,
      kQkvKPacked,
      kQkvGroupSize,
      kQkvFusedWidth,
      0u,
      0u};
  WGPUBuffer params_buffer = graph.create_params_buffer(params);
  WGPUBuffer bias_dummy = graph.create_scratch_buffer(4);

  const bool initially_active = is_qkv_bk64_live_m(fusion.max_m);
  const uint32_t workgroups =
      ((fusion.max_m + kQkvTile - 1u) / kQkvTile) * (kQkvFusedWidth / kQkvTile);
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = "q4gsw_qkv_bk64";
  descriptor.kernel_name = "linear_q4gsw_bk64_qkv";
  descriptor.bindings = {
      {output_q.buffer, 0u, output_q.nbytes},
      {output_k.buffer, 0u, output_k.nbytes},
      {output_v.buffer, 0u, output_v.nbytes},
      {input.buffer, 0u, input.nbytes},
      {fused_weight,
       0u,
       static_cast<uint64_t>(kQkvFusedWidth) * weight_row_bytes},
      {fused_scales,
       0u,
       static_cast<uint64_t>(kQkvNumGroups) * kQkvFusedWidth * sizeof(float)},
      {bias_dummy, 0u, 4u},
      {params_buffer, 0u, sizeof(QkvBk64Params)}};
  descriptor.grid = {
      initially_active ? workgroups : 0u, initially_active ? 1u : 0u};
  fusion.fused_dispatch = graph.add_compute_dispatch(descriptor);
  fusion.params_buffer = params_buffer;
}

void add_qkv_bk64_resize_hook(WebGPUGraph& graph, const QkvBk64Fusion& fusion) {
  const QkvBk64ResizeContext context = {
      fusion.input_id,
      {fusion.output_ids[0], fusion.output_ids[1], fusion.output_ids[2]},
      {fusion.separate_begin[0],
       fusion.separate_begin[1],
       fusion.separate_begin[2]},
      {fusion.separate_end[0], fusion.separate_end[1], fusion.separate_end[2]},
      fusion.fused_dispatch,
      fusion.max_m,
      fusion.params_buffer};
  resize_qkv_bk64(graph, context);
  graph.add_tensor_resize_hook(fusion.input_id, [context](WebGPUGraph& g) {
    resize_qkv_bk64(g, context);
  });
}

} // namespace executorch::backends::webgpu::passes
