/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/ops/mul/silu_mul_fused_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_qkv_bk64_wgsl.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

// vkgraph namespace is declared at global scope in the generated FlatBuffer
// header

namespace {

// Op name the AOT exporter emits for a prepacked constant (must match the
// serialized schema); compared in the prepack pre-scan below.
constexpr const char* kPrepackOpName = "et_vk.prepack.default";
constexpr const char* kQ4gswLinearOpName = "et_vk.linear_q4gsw.default";
constexpr size_t kQ4gswOutputArg = 5;
constexpr const char* kSigmoidOpName = "aten.sigmoid.default";
constexpr const char* kMulOpName = "aten.mul.Tensor";

struct SiluMulParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

struct SwiGluFusion {
  int common_input_id;
  int gate_id;
  int up_id;
  int sigmoid_id;
  int silu_id;
  int out_id;
  unsigned gate_op;
  unsigned sigmoid_op;
  unsigned mul1_op;
  unsigned mul2_op;
};

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

struct QkvBk64Fusion {
  int input_id = -1;
  int output_ids[3] = {-1, -1, -1};
  int weight_ids[3] = {-1, -1, -1};
  int scale_ids[3] = {-1, -1, -1};
  unsigned op_indices[3] = {0, 0, 0};
  size_t separate_begin[3] = {0, 0, 0};
  size_t separate_end[3] = {0, 0, 0};
  size_t fused_dispatch = SIZE_MAX;
  WGPUBuffer params_buffer = nullptr;
  uint32_t max_m = 0;
};

constexpr uint32_t kQkvQWidth = 2048u;
constexpr uint32_t kQkvKvWidth = 512u;
constexpr uint32_t kQkvFusedWidth = 3072u;
constexpr uint32_t kQkvK = 2048u;
constexpr uint32_t kQkvKPacked = 1024u;
constexpr uint32_t kQkvGroupSize = 64u;
constexpr uint32_t kQkvNumGroups = 32u;
constexpr uint32_t kQkvTile = 64u;

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

bool is_qkv_bk64_live_m(uint32_t m) {
  return m == 128u || m == 508u || m == 512u;
}

bool is_fp32_tensor(const WebGPUTensor& tensor) {
  if (tensor.is_int || tensor.elem_size != sizeof(float) ||
      tensor.buffer == nullptr) {
    return false;
  }
  const uint64_t numel = utils::numel_of(tensor.dims);
  return numel <= std::numeric_limits<size_t>::max() / sizeof(float) &&
      tensor.nbytes == static_cast<size_t>(numel) * sizeof(float);
}

uint32_t checked_silu_mul_numel(const std::vector<int64_t>& dims) {
  const uint64_t numel = utils::numel_of(dims);
  if (numel == 0 || numel > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("silu_mul_fused: element count out of range");
  }
  return static_cast<uint32_t>(numel);
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
  const uint32_t wg_size =
      utils::clamp_workgroup_size(graph.device(), kSiluMulFusedWorkgroupSizeX);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      graph.device(), num_elements, wg_size, "silu_mul_fused");

  SiluMulParams params = {num_elements, {0u, 0u, 0u}};
  WGPUBuffer uniform_buffer =
      graph.make_uniform_buffer(&params, sizeof(params));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kSiluMulFusedWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader =
      wgpuDeviceCreateShaderModule(graph.device(), &shader_desc);

  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl =
      wgpuDeviceCreateBindGroupLayout(graph.device(), &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(graph.device(), &pl_desc);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(graph.device(), &pipeline_desc);

  WGPUBindGroupEntry bg_entries[4] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = gate.buffer;
  bg_entries[0].size = gate.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = up.buffer;
  bg_entries[1].size = up.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out.buffer;
  bg_entries[2].size = out.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(SiluMulParams);
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group =
      wgpuDeviceCreateBindGroup(graph.device(), &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "silu_mul_fused",
       workgroup_count.y});

  graph.add_tensor_resize_hook(
      common_input_id,
      [gate_id, up_id, out_id, wg_size, dispatch_idx, uniform_buffer](
          WebGPUGraph& g) {
        const auto& gate_dims = g.cur_dims(gate_id);
        const auto& up_dims = g.cur_dims(up_id);
        if (gate_dims != up_dims) {
          throw std::runtime_error(
              "silu_mul_fused(resize): gate/up shape mismatch");
        }
        const uint32_t live_numel = checked_silu_mul_numel(gate_dims);
        const size_t live_nbytes =
            static_cast<size_t>(live_numel) * sizeof(float);
        if (g.get_tensor(gate_id).cur_nbytes != live_nbytes ||
            g.get_tensor(up_id).cur_nbytes != live_nbytes) {
          throw std::runtime_error(
              "silu_mul_fused(resize): gate/up byte-size mismatch");
        }
        g.set_cur_dims(out_id, gate_dims);
        SiluMulParams live_params = {live_numel, {0u, 0u, 0u}};
        wgpuQueueWriteBuffer(
            g.queue(), uniform_buffer, 0, &live_params, sizeof(live_params));
        const utils::WgCount live_workgroup_count =
            utils::compute_2d_workgroup_count(
                g.device(), live_numel, wg_size, "silu_mul_fused(resize)");
        auto& dispatch = g.dispatch_at(dispatch_idx);
        dispatch.workgroup_count_x = live_workgroup_count.x;
        dispatch.workgroup_count_y = live_workgroup_count.y;
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
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

  QkvBk64Params params = {
      fusion.max_m,
      kQkvFusedWidth,
      kQkvK,
      kQkvKPacked,
      kQkvGroupSize,
      kQkvFusedWidth,
      0u,
      0u};
  WGPUBuffer params_buffer = graph.make_uniform_buffer(&params, sizeof(params));
  WGPUBuffer bias_dummy = graph.create_scratch_buffer(4);

  WGPUBindGroupLayoutEntry layout_entries[8] = {};
  for (uint32_t i = 0; i < 3; i++) {
    layout_entries[i].binding = i;
    layout_entries[i].visibility = WGPUShaderStage_Compute;
    layout_entries[i].buffer.type = WGPUBufferBindingType_Storage;
  }
  for (uint32_t i = 3; i < 7; i++) {
    layout_entries[i].binding = i;
    layout_entries[i].visibility = WGPUShaderStage_Compute;
    layout_entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
  layout_entries[7].binding = 7;
  layout_entries[7].visibility = WGPUShaderStage_Compute;
  layout_entries[7].buffer.type = WGPUBufferBindingType_Uniform;
  WGPUBindGroupLayout layout =
      graph.get_or_create_bgl("q4gsw_qkv_bk64_8bind", layout_entries, 8);

  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {};
  pipeline_layout_desc.bindGroupLayoutCount = 1;
  pipeline_layout_desc.bindGroupLayouts = &layout;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(graph.device(), &pipeline_layout_desc);
  WGPUShaderModule shader =
      graph.get_or_create_shader("linear_q4gsw_bk64_qkv", kQ4gswQkvBk64WGSL);
  WGPUComputePipeline pipeline = graph.get_or_create_pipeline(
      "linear_q4gsw_bk64_qkv", shader, pipeline_layout);
  wgpuComputePipelineAddRef(pipeline);
  wgpuPipelineLayoutRelease(pipeline_layout);

  WGPUBindGroupEntry bind_entries[8] = {};
  const WebGPUTensor* outputs[3] = {&output_q, &output_k, &output_v};
  for (uint32_t i = 0; i < 3; i++) {
    bind_entries[i].binding = i;
    bind_entries[i].buffer = outputs[i]->buffer;
    bind_entries[i].size = outputs[i]->nbytes;
  }
  bind_entries[3].binding = 3;
  bind_entries[3].buffer = input.buffer;
  bind_entries[3].size = input.nbytes;
  bind_entries[4].binding = 4;
  bind_entries[4].buffer = fused_weight;
  bind_entries[4].size =
      static_cast<uint64_t>(kQkvFusedWidth) * weight_row_bytes;
  bind_entries[5].binding = 5;
  bind_entries[5].buffer = fused_scales;
  bind_entries[5].size =
      static_cast<uint64_t>(kQkvNumGroups) * kQkvFusedWidth * sizeof(float);
  bind_entries[6].binding = 6;
  bind_entries[6].buffer = bias_dummy;
  bind_entries[6].size = 4;
  bind_entries[7].binding = 7;
  bind_entries[7].buffer = params_buffer;
  bind_entries[7].size = sizeof(params);
  WGPUBindGroupDescriptor bind_group_desc = {};
  bind_group_desc.layout = layout;
  bind_group_desc.entryCount = 8;
  bind_group_desc.entries = bind_entries;
  WGPUBindGroup bind_group =
      wgpuDeviceCreateBindGroup(graph.device(), &bind_group_desc);

  const bool initially_active = is_qkv_bk64_live_m(fusion.max_m);
  const uint32_t workgroups =
      ((fusion.max_m + kQkvTile - 1u) / kQkvTile) * (kQkvFusedWidth / kQkvTile);
  fusion.fused_dispatch = graph.add_dispatch(
      {pipeline,
       bind_group,
       initially_active ? workgroups : 0u,
       "linear_q4gsw_bk64_qkv",
       initially_active ? 1u : 0u});
  fusion.params_buffer = params_buffer;
  graph.own_uniform_buffer(params_buffer);
}

void add_qkv_bk64_resize_hook(WebGPUGraph& graph, const QkvBk64Fusion& fusion) {
  const int input_id = fusion.input_id;
  const std::array<int, 3> output_ids = {
      fusion.output_ids[0], fusion.output_ids[1], fusion.output_ids[2]};
  const std::array<size_t, 3> separate_begin = {
      fusion.separate_begin[0],
      fusion.separate_begin[1],
      fusion.separate_begin[2]};
  const std::array<size_t, 3> separate_end = {
      fusion.separate_end[0], fusion.separate_end[1], fusion.separate_end[2]};
  const size_t fused_dispatch = fusion.fused_dispatch;
  const uint32_t max_m = fusion.max_m;
  WGPUBuffer params_buffer = fusion.params_buffer;
  auto resize = [input_id,
                 output_ids,
                 separate_begin,
                 separate_end,
                 fused_dispatch,
                 max_m,
                 params_buffer](WebGPUGraph& g) {
    const auto& input_dims = g.cur_dims(input_id);
    const uint64_t input_numel = utils::numel_of(input_dims);
    if (input_dims.empty() || input_numel % kQkvK != 0u) {
      throw std::runtime_error(
          "linear_q4gsw_bk64_qkv(resize): malformed input shape");
    }
    const uint64_t live_m = input_numel / kQkvK;
    if (live_m == 0u || live_m > max_m) {
      throw std::runtime_error(
          "linear_q4gsw_bk64_qkv(resize): live M out of range");
    }
    const uint32_t m = static_cast<uint32_t>(live_m);
    const uint32_t widths[3] = {kQkvQWidth, kQkvKvWidth, kQkvKvWidth};
    for (size_t i = 0; i < output_ids.size(); i++) {
      std::vector<int64_t> output_dims = input_dims;
      output_dims.back() = widths[i];
      g.set_cur_dims(output_ids[i], output_dims);
    }

    QkvBk64Params params = {
        m,
        kQkvFusedWidth,
        kQkvK,
        kQkvKPacked,
        kQkvGroupSize,
        kQkvFusedWidth,
        0u,
        0u};
    wgpuQueueWriteBuffer(g.queue(), params_buffer, 0, &params, sizeof(params));

    const bool use_fused = is_qkv_bk64_live_m(m);
    auto& fused = g.dispatch_at(fused_dispatch);
    fused.workgroup_count_x = use_fused
        ? ((m + kQkvTile - 1u) / kQkvTile) * (kQkvFusedWidth / kQkvTile)
        : 0u;
    fused.workgroup_count_y = use_fused ? 1u : 0u;
    // Only the fused path zeros the separate q/k/v dispatches. When !use_fused
    // they are left as-is: each separate projection's own resize hook is
    // registered on this same input_id and runs before this (last-registered)
    // hook, re-setting its live-M workgroup count every resize. So a prior
    // fused invocation's zeros are always overwritten before this hook runs.
    if (use_fused) {
      for (size_t member = 0; member < separate_begin.size(); member++) {
        for (size_t i = separate_begin[member]; i < separate_end[member]; i++) {
          auto& dispatch = g.dispatch_at(i);
          dispatch.workgroup_count_x = 0u;
          dispatch.workgroup_count_y = 0u;
        }
      }
    }
  };
  resize(graph);
  graph.add_tensor_resize_hook(input_id, std::move(resize));
}

size_t vk_datatype_size(vkgraph::VkDataType dtype) {
  switch (dtype) {
    case vkgraph::VkDataType::BOOL:
    case vkgraph::VkDataType::UINT8:
    case vkgraph::VkDataType::INT8:
      return 1;
    case vkgraph::VkDataType::FLOAT16:
      return 2;
    case vkgraph::VkDataType::INT32:
    case vkgraph::VkDataType::FLOAT32:
      return 4;
    case vkgraph::VkDataType::INT64:
    case vkgraph::VkDataType::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

bool vk_datatype_is_int(vkgraph::VkDataType dtype) {
  switch (dtype) {
    case vkgraph::VkDataType::BOOL:
    case vkgraph::VkDataType::UINT8:
    case vkgraph::VkDataType::INT8:
    case vkgraph::VkDataType::INT32:
    case vkgraph::VkDataType::INT64:
      return true;
    default:
      return false;
  }
}

// Normalize a possibly-negative dim against rank; throws (fail-loud) if OOR.
int normalize_dim(int dim, int rank, const char* op) {
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::runtime_error(
        std::string("WebGPU ") + op + ": dim out of range");
  }
  return dim;
}

} // namespace

WebGPUGraph::WebGPUGraph() = default;

WGPUBuffer WebGPUGraph::create_scratch_buffer(size_t nbytes) {
  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = nbytes > 0 ? nbytes : 4;
  buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = false;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
  scratch_buffers_.push_back(buffer);
  return buffer;
}

WGPUBuffer WebGPUGraph::acquire_scratch(size_t nbytes) {
  nbytes = nbytes > 0 ? nbytes : 4;
  // Best-fit reuse: smallest free slot with size in [nbytes, 2*nbytes] -- the
  // 2x cap stops a large Cmax-sized buffer from backing a tiny request. Never
  // reuse an in_use slot (co-live safety).
  ScratchSlot* best = nullptr;
  for (auto& s : scratch_pool_) {
    // s.size - nbytes (safe: s.size >= nbytes) avoids overflowing 2 * nbytes.
    if (!s.in_use && s.size >= nbytes && s.size - nbytes <= nbytes) {
      if (best == nullptr || s.size < best->size) {
        best = &s;
      }
    }
  }
  if (best != nullptr) {
    best->in_use = true;
    return best->buffer;
  }
  // None reusable -> create a new slot (freed in the dtor, like
  // scratch_buffers_).
  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = nbytes;
  buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = false;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
  scratch_pool_.push_back({buffer, nbytes, true});
  return buffer;
}

void WebGPUGraph::release_scratch(WGPUBuffer buffer) {
  if (!buffer) {
    return;
  }
  for (auto& s : scratch_pool_) {
    if (s.buffer == buffer) {
      s.in_use = false;
      return;
    }
  }
  // Not a pooled buffer -> no-op; the dtor frees it via scratch_buffers_.
}

WGPUBuffer WebGPUGraph::make_uniform_buffer(const void* data, size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &desc);
  void* mapped = wgpuBufferGetMappedRange(buffer, 0, size);
  std::memcpy(mapped, data, size);
  wgpuBufferUnmap(buffer);
  uniform_buffer_bytes_ += size;
  return buffer;
}

void WebGPUGraph::update_symints_from_inputs(
    const std::vector<InputData>& inputs) {
  for (const auto& src : symint_sources_) {
    int pos = -1;
    for (size_t i = 0; i < input_ids_.size(); i++) {
      if (input_ids_[i] == src.input_tensor_id) {
        pos = static_cast<int>(i);
        break;
      }
    }
    if (pos < 0 || pos >= static_cast<int>(inputs.size())) {
      throw std::runtime_error(
          "select_as_symint: source tensor is not a graph input");
    }
    // Live cur_dims: the source may be a dynamic-shape input.
    const auto& dims = tensors_[src.input_tensor_id].cur_dims;
    int dim = normalize_dim(
        src.dim, static_cast<int>(dims.size()), "select_as_symint");
    int index = src.index;
    if (index < 0) {
      index += static_cast<int>(dims[dim]);
    }
    if (index < 0 || index >= static_cast<int>(dims[dim])) {
      throw std::runtime_error("select_as_symint: index out of range");
    }
    int64_t numel = 1;
    for (int64_t d : dims) {
      numel *= d;
    }
    if (numel <= 0) {
      throw std::runtime_error("select_as_symint: empty input tensor");
    }
    int64_t stride = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < dims.size(); i++) {
      stride *= dims[i];
    }
    // Reads the [0,..,index,..,0] element; symint sources are scalar-ish.
    const int64_t offset = static_cast<int64_t>(index) * stride;
    const void* host = inputs[pos].data;
    // Interpret the HOST buffer by its scalar type, not the tensor's serialized
    // elem_size: copy_inputs narrows an int64 host input to an int32 buffer, so
    // elem_size (buffer-derived) would misread int64 host data as int32.
    int32_t val;
    if (inputs[pos].host_is_int64) {
      val = static_cast<int32_t>(static_cast<const int64_t*>(host)[offset]);
    } else {
      val = static_cast<const int32_t*>(host)[offset];
    }
    set_symint(src.symint_id, val);
  }
  // sym_size.int: SymInt = a tensor's live dim (cur_dims). Usually unused (ops
  // read cur_dims directly); for an intermediate source cur_dims is the build
  // max here (hooks run later in propagate_resize), which is fine while unused.
  for (const auto& s : symint_dim_sources_) {
    const auto& d = tensors_[s.tensor_id].cur_dims;
    int dim = normalize_dim(s.dim, static_cast<int>(d.size()), "sym_size");
    set_symint(s.symint_id, static_cast<int32_t>(d[dim]));
  }
}

void WebGPUGraph::set_symint(int id, int32_t val) {
  auto it = symints_.find(id);
  if (it == symints_.end()) {
    throw std::runtime_error("WebGPUGraph::set_symint: id is not a SymInt");
  }
  if (it->second.value != val) {
    it->second.value = val;
    wgpuQueueWriteBuffer(
        queue_, it->second.buffer, 0, &it->second.value, sizeof(int32_t));
    dirty_symints_.insert(id);
  }
}

void WebGPUGraph::set_cur_dims(
    int value_id,
    const std::vector<int64_t>& new_dims) {
  auto& t = tensors_[value_id];
  if (new_dims.size() != t.dims.size()) {
    throw std::runtime_error("WebGPU resize: tensor rank changed");
  }
  size_t numel = 1;
  for (size_t d = 0; d < new_dims.size(); d++) {
    // 0-sized dims unsupported: live shapes are always in [1, max] per dim.
    if (new_dims[d] <= 0) {
      throw std::runtime_error("WebGPU resize: new dim must be positive");
    }
    if (new_dims[d] > t.dims[d]) {
      throw std::runtime_error(
          "WebGPU resize: new dim exceeds the max (serialized) allocation");
    }
    numel *= static_cast<size_t>(new_dims[d]);
  }
  const size_t new_nbytes = numel * t.elem_size;
  if (t.cur_dims != new_dims) {
    t.cur_dims = new_dims;
    t.cur_nbytes = new_nbytes;
    dirty_tensors_.insert(value_id);
  }
}

void WebGPUGraph::resize_input(
    int value_id,
    const std::vector<int64_t>& new_dims) {
  if (std::find(input_ids_.begin(), input_ids_.end(), value_id) ==
      input_ids_.end()) {
    throw std::runtime_error(
        "WebGPUGraph::resize_input: value_id is not a graph input");
  }
  set_cur_dims(value_id, new_dims);
}

void WebGPUGraph::propagate_resize() {
  if (dirty_symints_.empty() && dirty_tensors_.empty()) {
    return;
  }
  // Hooks fire in registration (topological) order: operands update first.
  for (auto& hook : resize_hooks_) {
    if (dirty_symints_.count(hook.symint_id) != 0) {
      hook.fn(*this);
    }
  }
  dirty_symints_.clear();
  // Tensor hooks: bounded fixpoint. A hook may dirty its output (cascading to a
  // consumer); each pass handles the currently-dirty set. A forward DAG
  // converges in <= depth passes (set_cur_dims re-dirties only on a change).
  for (size_t pass = 0;
       !dirty_tensors_.empty() && pass <= tensor_resize_hooks_.size();
       pass++) {
    std::unordered_set<int> processing;
    processing.swap(dirty_tensors_);
    for (auto& hook : tensor_resize_hooks_) {
      if (processing.count(hook.trigger_tensor_id) != 0) {
        hook.fn(*this);
      }
    }
  }
  if (!dirty_tensors_.empty()) {
    throw std::runtime_error(
        "WebGPU resize: tensor resize hooks did not converge");
  }
  // Tensor hooks must not set_symint (dirty_symints_ already drained above).
  if (!dirty_symints_.empty()) {
    throw std::runtime_error(
        "WebGPU resize: a tensor resize hook set a SymInt; not supported");
  }
}

WebGPUGraph::~WebGPUGraph() {
  for (size_t i = 0; i < tensors_.size(); i++) {
    if (tensors_[i].buffer &&
        (i >= tensor_mem_obj_ids_.size() || tensor_mem_obj_ids_[i] < 0)) {
      wgpuBufferRelease(tensors_[i].buffer);
    }
  }
  for (auto& buf : shared_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& buf : scratch_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& s : scratch_pool_) {
    if (s.buffer) {
      wgpuBufferRelease(s.buffer);
    }
  }
  for (auto& buf : owned_uniform_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& kv : symints_) {
    if (kv.second.buffer) {
      wgpuBufferRelease(kv.second.buffer);
    }
  }
  for (auto& buf : output_staging_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& d : dispatches_) {
    if (d.pipeline) {
      wgpuComputePipelineRelease(d.pipeline);
    }
    if (d.bind_group) {
      wgpuBindGroupRelease(d.bind_group);
    }
  }
  for (auto& [_, shader] : shader_cache_) {
    if (shader) {
      wgpuShaderModuleRelease(shader);
    }
  }
  for (auto& [_, pipeline] : pipeline_cache_) {
    if (pipeline) {
      wgpuComputePipelineRelease(pipeline);
    }
  }
  for (auto& [_, bgl] : bgl_cache_) {
    if (bgl) {
      wgpuBindGroupLayoutRelease(bgl);
    }
  }
}

void WebGPUGraph::build(
    const void* flatbuffer_data,
    const uint8_t* constant_data,
    const executorch::runtime::NamedDataMap* named_data_map,
    bool f16_kv_cache,
    bool f16_accumulate_gemm,
    int sdpa_query_tile) {
  if (!device_) {
    auto* ctx = get_default_webgpu_context();
    if (ctx) {
      device_ = ctx->device;
      instance_ = ctx->instance;
    }
  }
  if (!device_) {
    throw std::runtime_error(
        "WebGPU device not available. "
        "Call set_default_webgpu_context() before loading.");
  }
  queue_ = wgpuDeviceGetQueue(device_);

  const auto* graph = vkgraph::GetVkGraph(flatbuffer_data);

  // .pte byte sources for prepack-time constant materialization (build-only).
  constant_data_ = constant_data;
  named_data_map_ = named_data_map;

  // f16 KV cache (runtime opt-in): store K/V caches as f16 iff the opt-in is
  // set AND the device negotiated shader-f16 (fail-closed).
  const WebGPUContext* kv_ctx = get_default_webgpu_context();
  kv_f16_ = f16_kv_cache && (kv_ctx != nullptr && kv_ctx->shader_f16_supported);

  // f16-accumulate q4gsw steel prefill GEMM (runtime opt-in). QuantizedLinear
  // additionally gates the kernel on the negotiated shader-f16 feature.
  f16_accumulate_gemm_ = f16_accumulate_gemm;

  // SDPA query-tile selector (runtime opt-in); 0 = geometry default (Q16),
  // 32 = Q32 candidate. Read at the SDPA op-lowering selection site.
  sdpa_query_tile_ = sdpa_query_tile;

  // Phase 1: Create all values
  const auto* values = graph->values();
  const int num_vals = values ? values->size() : 0;
  value_types_.resize(num_vals, ValueType::Null);
  tensors_.resize(num_vals);
  tensor_mem_obj_ids_.resize(num_vals, -1);
  ints_.resize(num_vals, 0);
  int_lists_.resize(num_vals);
  value_lists_.resize(num_vals);
  doubles_.resize(num_vals, 0.0);
  bools_.resize(num_vals, false);

  // Pre-scan the op chain: a constant may be DEFERRED (no eager GPU buffer; the
  // prepack node materializes it once) only if it is a prepack source AND never
  // a direct arg of a non-prepack op. ValueList args are expanded so a constant
  // reached through a list still counts as a direct use.
  std::unordered_set<int> prepack_src_ids;
  std::unordered_set<int> direct_use_ids;
  const auto* chain_prescan = graph->chain();
  if (chain_prescan) {
    for (unsigned ci = 0; ci < chain_prescan->size(); ci++) {
      const auto* oc = chain_prescan->Get(ci);
      const bool is_prepack = oc->name()->str() == kPrepackOpName;
      const auto* a = oc->args();
      if (!a) {
        continue;
      }
      if (oc->name()->str() == "sym_size.int" && a->size() >= 3 && values) {
        const auto* out = values->Get(a->Get(2));
        if (out && out->value_type() == vkgraph::GraphTypes::SymInt) {
          dynamic_tensor_ids_.insert(static_cast<int>(a->Get(0)));
        }
      }
      // f16 KV: tag sdpa K/V cache values (args[3],[4]) for half-size alloc.
      // Inert unless kv_f16_ (runtime opt-in) is set.
      if (kv_f16_ && a->size() > 4 &&
          oc->name()->str() == "sdpa_with_kv_cache.default") {
        kv_cache_ids_.insert(static_cast<int>(a->Get(3)));
        kv_cache_ids_.insert(static_cast<int>(a->Get(4)));
      }
      for (unsigned j = 0; j < a->size(); j++) {
        int id = static_cast<int>(a->Get(j));
        if (is_prepack && j == 0) {
          prepack_src_ids.insert(id);
        } else if (!is_prepack) {
          direct_use_ids.insert(id);
          const auto* v = values ? values->Get(id) : nullptr;
          if (v && v->value_type() == vkgraph::GraphTypes::ValueList) {
            const auto* items = v->value_as_ValueList()->items();
            if (items) {
              for (unsigned k = 0; k < items->size(); k++) {
                direct_use_ids.insert(static_cast<int>(items->Get(k)));
              }
            }
          }
        }
      }
    }
  }

  // f16 KV defensive guard: fail loud if a non-sdpa op reads an f16 cache.
  // Inert unless kv_f16_ (runtime opt-in) is set.
  if (kv_f16_ && !kv_cache_ids_.empty() && chain_prescan) {
    for (unsigned ci = 0; ci < chain_prescan->size(); ci++) {
      const auto* oc = chain_prescan->Get(ci);
      const std::string nm = oc->name()->str();
      if (nm == "sdpa_with_kv_cache.default" || nm == kPrepackOpName) {
        continue;
      }
      const auto* a = oc->args();
      if (!a) {
        continue;
      }
      for (unsigned j = 0; j < a->size(); j++) {
        if (kv_cache_ids_.count(static_cast<int>(a->Get(j))) != 0) {
          throw std::runtime_error(
              "WebGPU f16 KV: cache tensor consumed by non-sdpa op '" + nm +
              "' would misread the f16 buffer");
        }
      }
    }
  }

  for (int i = 0; i < num_vals; i++) {
    const auto* val = values->Get(i);
    if (!val || val->value_type() == vkgraph::GraphTypes::NONE) {
      value_types_[i] = ValueType::Null;
      continue;
    }

    switch (val->value_type()) {
      case vkgraph::GraphTypes::VkTensor: {
        value_types_[i] = ValueType::Tensor;
        const auto* vk_tensor = val->value_as_VkTensor();
        auto& tensor = tensors_[i];

        const auto* dims = vk_tensor->dims();
        size_t numel = 1;
        if (dims) {
          for (unsigned j = 0; j < dims->size(); j++) {
            tensor.dims.push_back(static_cast<int64_t>(dims->Get(j)));
            numel *= dims->Get(j);
          }
        }
        tensor.elem_size = vk_datatype_size(vk_tensor->datatype());
        tensor.is_int = vk_datatype_is_int(vk_tensor->datatype());
        tensor.nbytes = numel * tensor.elem_size;
        // Live dims start == max (serialized upper bound); resize_input shrinks
        // them per call. Static graphs keep cur == max forever.
        tensor.cur_dims = tensor.dims;
        tensor.cur_nbytes = tensor.nbytes;

        // f16 KV cache: dedicated half-size array<f16> buffer. WebGPU
        // zero-initializes freshly-created buffers, so no explicit clear is
        // needed. Inert unless kv_f16_ (runtime opt-in) is set.
        if (kv_f16_ && kv_cache_ids_.count(i) != 0) {
          tensor.elem_size = 2;
          tensor.nbytes = numel * 2;
          tensor.cur_nbytes = tensor.nbytes;
          tensor_mem_obj_ids_[i] = -1;
          WGPUBufferDescriptor buf_desc = {};
          buf_desc.size = std::max(tensor.nbytes, size_t(4));
          buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
          buf_desc.mappedAtCreation = false;
          tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
          break;
        }

        int constant_id = vk_tensor->constant_id();
        int mem_obj_id = vk_tensor->mem_obj_id();

        // Constants are dedicated. Every constant is recorded as a
        // ConstantSource and materialized via materialize_constant (one
        // CPU->GPU write); a constant consumed ONLY via prepack is deferred
        // (no eager buffer -- its prepack node performs that one write).
        if (constant_id >= 0 || mem_obj_id < 0) {
          tensor_mem_obj_ids_[i] = -1;

          if (constant_id >= 0) {
            const auto* constants = graph->constants();
            if (!constants ||
                constant_id >= static_cast<int>(constants->size())) {
              throw std::runtime_error(
                  "WebGPU: constant_id set but the constants table is missing "
                  "or the id is out of range");
            }
            const auto* vk_bytes = constants->Get(constant_id);
            ConstantSource cs;
            cs.nbytes = tensor.nbytes;
            if (vk_bytes->offset() != UINT64_MAX) {
              cs.inline_offset = vk_bytes->offset();
            } else if (vk_bytes->named_key() != nullptr) {
              cs.named_key = vk_bytes->named_key()->str();
            } else {
              throw std::runtime_error(
                  "WebGPU: constant has no inline offset and no named-data key");
            }
            constant_sources_[i] = std::move(cs);
          }

          // Defer constants consumed solely via prepack: skip the eager buffer.
          const bool defer = constant_id >= 0 &&
              prepack_src_ids.count(i) != 0 && direct_use_ids.count(i) == 0;
          if (!defer) {
            WGPUBufferDescriptor buf_desc = {};
            buf_desc.size = std::max(tensor.nbytes, size_t(4));
            buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                WGPUBufferUsage_CopySrc;
            buf_desc.mappedAtCreation = false;
            tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);

            // Same single CPU->GPU write the prepack node uses (no
            // duplication).
            if (constant_id >= 0) {
              materialize_constant(i, tensor.buffer);
            }
          }
        } else {
          // Shared buffer: track required size, defer allocation to pass 2
          tensor_mem_obj_ids_[i] = mem_obj_id;
          size_t id = static_cast<size_t>(mem_obj_id);
          if (id >= shared_buffer_sizes_.size()) {
            shared_buffer_sizes_.resize(id + 1, 0);
          }
          shared_buffer_sizes_[id] =
              std::max(shared_buffer_sizes_[id], tensor.nbytes);
        }
        break;
      }
      case vkgraph::GraphTypes::Int: {
        value_types_[i] = ValueType::Int;
        ints_[i] = val->value_as_Int()->int_val();
        break;
      }
      case vkgraph::GraphTypes::IntList: {
        value_types_[i] = ValueType::IntList;
        const auto* items = val->value_as_IntList()->items();
        if (items) {
          int_lists_[i].assign(items->cbegin(), items->cend());
        }
        break;
      }
      case vkgraph::GraphTypes::ValueList: {
        value_types_[i] = ValueType::ValueList;
        const auto* items = val->value_as_ValueList()->items();
        if (items) {
          value_lists_[i].reserve(items->size());
          for (unsigned j = 0; j < items->size(); j++) {
            value_lists_[i].push_back(static_cast<int>(items->Get(j)));
          }
        }
        break;
      }
      case vkgraph::GraphTypes::Double: {
        value_types_[i] = ValueType::Double;
        doubles_[i] = val->value_as_Double()->double_val();
        break;
      }
      case vkgraph::GraphTypes::Bool: {
        value_types_[i] = ValueType::Bool;
        bools_[i] = val->value_as_Bool()->bool_val();
        break;
      }
      case vkgraph::GraphTypes::SymInt: {
        // Live scalar: small Uniform buffer the CPU rewrites per execute.
        value_types_[i] = ValueType::SymInt;
        SymIntSlot slot;
        slot.value = static_cast<int32_t>(val->value_as_SymInt()->value());
        // 16B matches the backend uniform-struct alignment; int32 in first 4.
        constexpr size_t kSymIntUniformBytes = 16;
        WGPUBufferDescriptor d = {};
        d.size = kSymIntUniformBytes;
        d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        d.mappedAtCreation = true;
        slot.buffer = wgpuDeviceCreateBuffer(device_, &d);
        void* mapped =
            wgpuBufferGetMappedRange(slot.buffer, 0, kSymIntUniformBytes);
        std::memset(mapped, 0, kSymIntUniformBytes);
        std::memcpy(mapped, &slot.value, sizeof(int32_t));
        wgpuBufferUnmap(slot.buffer);
        symints_[i] = slot;
        add_uniform_buffer_bytes(kSymIntUniformBytes);
        break;
      }
      default:
        value_types_[i] = ValueType::Null;
        break;
    }
  }

  // Allocate shared buffers and assign to tensors
  shared_buffers_.resize(shared_buffer_sizes_.size(), nullptr);
  for (size_t id = 0; id < shared_buffer_sizes_.size(); id++) {
    WGPUBufferDescriptor buf_desc = {};
    buf_desc.size = std::max(shared_buffer_sizes_[id], size_t(4));
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
        WGPUBufferUsage_CopySrc;
    buf_desc.mappedAtCreation = false;
    shared_buffers_[id] = wgpuDeviceCreateBuffer(device_, &buf_desc);
  }
  for (int i = 0; i < num_vals; i++) {
    int mid = tensor_mem_obj_ids_[i];
    if (mid >= 0) {
      tensors_[i].buffer = shared_buffers_[mid];
    }
  }

  // Phase 2: Record input and output IDs
  const auto* fb_input_ids = graph->input_ids();
  if (fb_input_ids) {
    for (unsigned i = 0; i < fb_input_ids->size(); i++) {
      input_ids_.push_back(static_cast<int>(fb_input_ids->Get(i)));
    }
  }
  const auto* fb_output_ids = graph->output_ids();
  if (fb_output_ids) {
    for (unsigned i = 0; i < fb_output_ids->size(); i++) {
      int oid = static_cast<int>(fb_output_ids->Get(i));
      output_ids_.push_back(oid);

      // Create staging buffer for output readback
      WGPUBufferDescriptor staging_desc = {};
      staging_desc.size = std::max(tensors_[oid].nbytes, size_t(4));
      staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
      staging_desc.mappedAtCreation = false;
      output_staging_buffers_.push_back(
          wgpuDeviceCreateBuffer(device_, &staging_desc));
    }
  }

  for (size_t i = 0; i < output_ids_.size(); i++) {
    int oid = output_ids_[i];
    output_copies_.push_back(
        {tensors_[oid].buffer,
         output_staging_buffers_[i],
         tensors_[oid].nbytes});
  }

  std::vector<SwiGluFusion> swiglu_fusions;
  std::unordered_map<unsigned, size_t> swiglu_gate_producers;
  std::unordered_map<unsigned, size_t> swiglu_anchors;
  std::unordered_set<unsigned> swiglu_skipped_ops;

  std::vector<QkvBk64Fusion> qkv_fusions;
  std::unordered_map<unsigned, size_t> qkv_first_ops;
  std::unordered_map<unsigned, size_t> qkv_last_ops;
  std::unordered_map<unsigned, size_t> qkv_member_ops;

  const auto* chain = graph->chain();
  if (chain && qkv_bk64_device_supported(device_)) {
    std::unordered_map<int, std::vector<unsigned>> q4_ops_by_input;
    std::vector<int> input_order;
    for (unsigned i = 0; i < chain->size(); i++) {
      const auto* op = chain->Get(i);
      const auto* args = op->args();
      if (op->name()->str() != kQ4gswLinearOpName || !args ||
          args->size() != 6) {
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
    auto is_graph_output = [&](int id) {
      return std::find(output_ids_.begin(), output_ids_.end(), id) !=
          output_ids_.end();
    };
    for (int input_id : input_order) {
      const auto& ops = q4_ops_by_input.at(input_id);
      if (ops.size() != 3 || input_id < 0 || input_id >= num_vals ||
          value_types_[input_id] != ValueType::Tensor) {
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
            value_types_[group_size_id] == ValueType::Int &&
            ints_[group_size_id] == kQkvGroupSize && bias_id >= 0 &&
            bias_id < num_vals && value_types_[bias_id] == ValueType::Null;
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
            value_types_[id] == ValueType::Tensor &&
            constant_sources_.count(id) != 0 && tensors_[id].buffer != nullptr;
      }
      if (!direct_constants) {
        continue;
      }

      const std::unordered_set<int> distinct_outputs = {
          fusion.output_ids[0], fusion.output_ids[1], fusion.output_ids[2]};
      bool outputs_ok = distinct_outputs.size() == 3;
      for (int id : fusion.output_ids) {
        outputs_ok = outputs_ok && id >= 0 && id < num_vals &&
            value_types_[id] == ValueType::Tensor &&
            tensor_mem_obj_ids_[id] >= 0 && !is_graph_output(id) &&
            is_fp32_tensor(tensors_[id]);
      }
      if (!outputs_ok) {
        continue;
      }

      const auto& input = tensors_[input_id];
      if (!is_fp32_tensor(input) || input.dims.empty() ||
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
        const auto& weight = tensors_[fusion.weight_ids[member]];
        const auto& scale = tensors_[fusion.scale_ids[member]];
        const auto& output = tensors_[fusion.output_ids[member]];
        exact_geometry = exact_geometry && weight.dims.size() == 2 &&
            weight.dims[0] == widths[member] && weight.dims[1] == kQkvKPacked &&
            weight.nbytes ==
                static_cast<size_t>(widths[member]) * kQkvKPacked &&
            scale.dims.size() == 2 && scale.dims[0] == kQkvNumGroups &&
            scale.dims[1] == widths[member] && is_fp32_tensor(scale) &&
            output.dims.size() == input.dims.size() &&
            std::equal(input.dims.begin(),
                       input.dims.end() - 1,
                       output.dims.begin()) &&
            output.dims.back() == widths[member] &&
            utils::numel_of(output.dims) ==
                static_cast<uint64_t>(fusion.max_m) * widths[member];
      }
      if (!exact_geometry) {
        continue;
      }

      const size_t fusion_index = qkv_fusions.size();
      qkv_fusions.push_back(fusion);
      qkv_first_ops[ops[0]] = fusion_index;
      qkv_last_ops[ops[2]] = fusion_index;
      for (unsigned op : ops) {
        qkv_member_ops[op] = fusion_index;
      }
    }
  }

  // Detect only the exact q4 gate/up MLP pattern. Full-chain occurrence and
  // definition counts make every folded intermediate private to the pattern.
  if (chain) {
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
      if (value_types_[id] == ValueType::ValueList) {
        for (int member : value_lists_[id]) {
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

    auto is_graph_output = [&](int id) {
      return std::find(output_ids_.begin(), output_ids_.end(), id) !=
          output_ids_.end();
    };
    std::unordered_set<unsigned> claimed_ops;
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
              value_types_[id] != ValueType::Tensor || definitions[id] != 1) {
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
            is_graph_output(silu_id) || tensor_mem_obj_ids_[gate_id] < 0) {
          return;
        }

        const auto& gate = tensors_[gate_id];
        const auto& up = tensors_[up_id];
        const auto& sigmoid = tensors_[sigmoid_id];
        const auto& silu = tensors_[silu_id];
        const auto& out = tensors_[out_id];
        if (!is_fp32_tensor(gate) || !is_fp32_tensor(up) ||
            !is_fp32_tensor(sigmoid) || !is_fp32_tensor(silu) ||
            !is_fp32_tensor(out) || gate.dims != up.dims ||
            gate.dims != sigmoid.dims || gate.dims != silu.dims ||
            gate.dims != out.dims || gate.nbytes != up.nbytes ||
            gate.nbytes != sigmoid.nbytes || gate.nbytes != silu.nbytes ||
            gate.nbytes != out.nbytes || up.buffer == out.buffer) {
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
      const size_t fusion_idx = swiglu_fusions.size();
      swiglu_fusions.push_back(fusion);
      swiglu_gate_producers[fusion.gate_op] = fusion_idx;
      swiglu_anchors[fusion.mul2_op] = fusion_idx;
      swiglu_skipped_ops.insert(fusion.sigmoid_op);
      swiglu_skipped_ops.insert(fusion.mul1_op);
      // mul2_op is the fusion anchor: its Phase-3 branch emits the fused
      // dispatch and continues before the skipped-ops check, so it needs no
      // swiglu_skipped_ops entry.
      for (unsigned op : pattern_ops) {
        claimed_ops.insert(op);
      }
    }
  }

  // Phase 3: Build operator dispatch chain
  if (chain) {
    for (unsigned i = 0; i < chain->size(); i++) {
      const auto* op_call = chain->Get(i);
      std::string op_name = op_call->name()->str();

      if (!webgpu_operator_registry().has_op(op_name)) {
        throw std::runtime_error("WebGPU backend: unsupported op: " + op_name);
      }

      const auto* fb_args = op_call->args();
      std::vector<int> args;
      if (fb_args) {
        for (unsigned j = 0; j < fb_args->size(); j++) {
          args.push_back(static_cast<int>(fb_args->Get(j)));
        }
      }

      const auto gate_it = swiglu_gate_producers.find(i);
      if (gate_it != swiglu_gate_producers.end()) {
        const int gate_id = swiglu_fusions[gate_it->second].gate_id;
        tensors_[gate_id].buffer = acquire_scratch(tensors_[gate_id].nbytes);
      }
      const auto anchor_it = swiglu_anchors.find(i);
      if (anchor_it != swiglu_anchors.end()) {
        const SwiGluFusion& fusion = swiglu_fusions[anchor_it->second];
        add_silu_mul_fused_dispatch(
            *this,
            fusion.common_input_id,
            fusion.gate_id,
            fusion.up_id,
            fusion.out_id);
        release_scratch(tensors_[fusion.gate_id].buffer);
        continue;
      }
      if (swiglu_skipped_ops.count(i) != 0) {
        continue;
      }

      const auto qkv_first = qkv_first_ops.find(i);
      if (qkv_first != qkv_first_ops.end()) {
        QkvBk64Fusion& fusion = qkv_fusions[qkv_first->second];
        for (int output_id : fusion.output_ids) {
          tensors_[output_id].buffer =
              create_scratch_buffer(tensors_[output_id].nbytes);
        }
      }

      const size_t dispatch_begin = dispatches_.size();
      webgpu_operator_registry().get_op_fn(op_name)(*this, args);
      const size_t dispatch_end = dispatches_.size();

      const auto qkv_member = qkv_member_ops.find(i);
      if (qkv_member != qkv_member_ops.end()) {
        QkvBk64Fusion& fusion = qkv_fusions[qkv_member->second];
        size_t member = 0;
        while (member < 3 && fusion.op_indices[member] != i) {
          member++;
        }
        if (member == 3 || dispatch_end <= dispatch_begin) {
          throw std::runtime_error(
              "linear_q4gsw_bk64_qkv: malformed member dispatch range");
        }
        fusion.separate_begin[member] = dispatch_begin;
        fusion.separate_end[member] = dispatch_end;
        if (member == 0) {
          add_qkv_bk64_dispatch(*this, fusion);
        }
      }
      const auto qkv_last = qkv_last_ops.find(i);
      if (qkv_last != qkv_last_ops.end()) {
        add_qkv_bk64_resize_hook(*this, qkv_fusions[qkv_last->second]);
      }

      if (i + 1 == chain->size() && op_name == kQ4gswLinearOpName &&
          args.size() > kQ4gswOutputArg && dispatch_end > dispatch_begin) {
        const int output_id = args[kQ4gswOutputArg];
        const auto output_it =
            std::find(output_ids_.begin(), output_ids_.end(), output_id);
        if (output_it != output_ids_.end() &&
            std::count(output_ids_.begin(), output_ids_.end(), output_id) ==
                1) {
          suppressible_outputs_.push_back(
              {output_id,
               static_cast<size_t>(output_it - output_ids_.begin()),
               dispatch_begin,
               dispatch_end});
        }
      }
    }
  }

  // Prepack nodes (Phase 3) materialized their constants directly into the
  // consumer buffers via materialize_constant; no separate copy pass needed.
  // The .pte bytes are freed right after build() returns (WebGPUBackend
  // processed->Free()), so clear the build-only source pointers.
  constant_data_ = nullptr;
  named_data_map_ = nullptr;
}

void WebGPUGraph::materialize_constant(int const_value_id, WGPUBuffer dst) {
  auto it = constant_sources_.find(const_value_id);
  if (it == constant_sources_.end()) {
    throw std::runtime_error(
        "WebGPU: no source recorded for constant id " +
        std::to_string(const_value_id));
  }
  const ConstantSource& cs = it->second;
  if (cs.nbytes == 0) {
    return;
  }
  if (cs.inline_offset != UINT64_MAX) {
    if (constant_data_ == nullptr) {
      throw std::runtime_error("WebGPU: inline constant data is null");
    }
    wgpuQueueWriteBuffer(
        queue_, dst, 0, constant_data_ + cs.inline_offset, cs.nbytes);
  } else if (!cs.named_key.empty() && named_data_map_ != nullptr) {
    auto buf = named_data_map_->get_data(cs.named_key.c_str());
    if (!buf.ok()) {
      throw std::runtime_error(
          "WebGPU: named constant '" + cs.named_key + "' not found");
    }
    if (buf->size() < cs.nbytes) {
      throw std::runtime_error(
          "WebGPU: named constant '" + cs.named_key + "' undersized");
    }
    wgpuQueueWriteBuffer(queue_, dst, 0, buf->data(), cs.nbytes);
    buf->Free();
  } else {
    throw std::runtime_error("WebGPU: constant has no source");
  }
}

WGPUShaderModule WebGPUGraph::get_or_create_shader(
    const std::string& key,
    const char* wgsl_source) {
  auto it = shader_cache_.find(key);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl_source, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device_, &shader_desc);

  shader_cache_[key] = shader;
  return shader;
}

WGPUComputePipeline WebGPUGraph::get_or_create_pipeline(
    const std::string& key,
    WGPUShaderModule shader,
    WGPUPipelineLayout layout) {
  auto it = pipeline_cache_.find(key);
  if (it != pipeline_cache_.end()) {
    return it->second;
  }

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device_, &pipeline_desc);

  pipeline_cache_[key] = pipeline;
  return pipeline;
}

WGPUBindGroupLayout WebGPUGraph::get_or_create_bgl(
    const std::string& key,
    const WGPUBindGroupLayoutEntry* entries,
    uint32_t count) {
  auto it = bgl_cache_.find(key);
  if (it != bgl_cache_.end()) {
    return it->second;
  }

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = count;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device_, &bgl_desc);

  bgl_cache_[key] = bgl;
  return bgl;
}

void WebGPUGraph::copy_inputs(const std::vector<InputData>& inputs) {
  for (size_t i = 0; i < inputs.size() && i < input_ids_.size(); i++) {
    const InputData& in = inputs[i];
    if (in.nbytes == 0) {
      continue;
    }
    int tid = input_ids_[i];
    const auto& tensor = tensors_[tid];
    // Upload only the live (cur) bytes, not the max allocation; cur_nbytes ==
    // nbytes on a static graph, so this is byte-identical there.
    const size_t live_nbytes = tensor.cur_nbytes;

    // Fast path: host and GPU element types match byte-for-byte.
    if (in.nbytes == live_nbytes) {
      wgpuQueueWriteBuffer(queue_, tensor.buffer, 0, in.data, live_nbytes);
      continue;
    }

    // Narrow int64 host indices into the int32 buffer (mirrors Vulkan).
    const bool buffer_is_int32 = tensor.is_int && tensor.elem_size == 4;
    if (in.host_is_int64 && buffer_is_int32 && in.nbytes == live_nbytes * 2) {
      const size_t numel = live_nbytes / 4;
      const int64_t* src = static_cast<const int64_t*>(in.data);
      std::vector<int32_t> narrowed(numel);
      for (size_t e = 0; e < numel; e++) {
#ifndef NDEBUG
        // Index tensors (tokens/positions) are far below int32 range in
        // practice; assert in debug that the narrowing is lossless.
        if (static_cast<int32_t>(src[e]) != src[e]) {
          throw std::runtime_error("WebGPU: int64 index overflows int32");
        }
#endif
        narrowed[e] = static_cast<int32_t>(src[e]);
      }
      wgpuQueueWriteBuffer(
          queue_, tensor.buffer, 0, narrowed.data(), live_nbytes);
      continue;
    }

    const bool buffer_is_fp16 = !tensor.is_int && tensor.elem_size == 2;
    // Require an explicit fp32 host dtype, not merely "not int64": inferring
    // the narrow from the 2:1 byte ratio alone would silently reinterpret a
    // same-sized non-fp32 host buffer (e.g. a stale int32) as fp32.
    if (in.host_is_fp32 && buffer_is_fp16 && in.nbytes == live_nbytes * 2) {
      const size_t numel = live_nbytes / sizeof(uint16_t);
      const float* src = static_cast<const float*>(in.data);
      std::vector<executorch::runtime::etensor::Half> narrowed(numel);
      for (size_t e = 0; e < numel; e++) {
        narrowed[e] = executorch::runtime::etensor::Half(src[e]);
      }
      wgpuQueueWriteBuffer(
          queue_, tensor.buffer, 0, narrowed.data(), live_nbytes);
      continue;
    }

    throw std::runtime_error(
        "WebGPU: unsupported input copy for input " + std::to_string(i) +
        " (host " + std::to_string(in.nbytes) + " bytes" +
        (in.host_is_int64 ? " int64" : "") + " vs buffer " +
        std::to_string(live_nbytes) + " bytes)");
  }
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
// Profiling/attestation only; never compiled into a production build. Written
// during WebGPUGraph::execute without synchronization: the attestation
// harnesses that read them run one graph on one thread, so no atomics or
// locking are needed. To support concurrent profiled execution, make these
// per-instance behind a whole-record mutex (per-field atomics would not cover
// the conflict check's read-modify-write across both globals).
uint32_t g_last_route_mask = 0;
uint32_t g_last_route_conflict_count = 0;
#endif // WGPU_BACKEND_ENABLE_PROFILING

namespace {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
constexpr uint32_t kRoutePrefill = 1u << 0;
constexpr uint32_t kRouteK16 = 1u << 1;
constexpr uint32_t kRouteMaterializedAttention = 1u << 2;
constexpr uint32_t kRouteT0Steel = 1u << 3;
constexpr uint32_t kRouteT1Bk64 = 1u << 4;
constexpr uint32_t kRouteT1Bk64Qkv = 1u << 5;
constexpr uint32_t kRouteT2PairedGateUp = 1u << 6;
constexpr uint32_t kRouteFusedSwiGlu = 1u << 7;
constexpr uint32_t kRouteGenericFallback = 1u << 8;
// bit 1u << 9 is intentionally reserved (a retired route) and left unused.
constexpr uint32_t kRouteFlashDecoding = 1u << 10;
constexpr uint32_t kRouteK16CausalBound = 1u << 11;
constexpr uint32_t kRouteBicolSubgroup = 1u << 12;
#endif // WGPU_BACKEND_ENABLE_PROFILING

// Bench gate: compiled out unless WGPU_BACKEND_ENABLE_PROFILING; then the
// WEBGPU_TIMESTAMP_QUERY env var enables per-pass GPU timestamp queries.
bool should_timestamp_query() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  return std::getenv("WEBGPU_TIMESTAMP_QUERY") != nullptr;
#else
  return false;
#endif
}
} // namespace

#ifdef WGPU_BACKEND_ENABLE_PROFILING
void WebGPUGraph::record_active_route(const std::string& kernel_name) {
  uint32_t bits = 0;
  if (kernel_name.rfind("sdpa_streaming_attention_", 0) == 0 &&
      kernel_name.find("k16_causal_bound") != std::string::npos) {
    // llama + qwen3 (Q16/Q32) streaming causal-bound kernels; the
    // sdpa_streaming_attention_ prefix guards against an unrelated future
    // kernel whose label merely contains the k16_causal_bound substring.
    bits = kRoutePrefill | kRouteK16CausalBound;
  } else if (kernel_name == "sdpa_streaming_attention_k16") {
    bits = kRoutePrefill | kRouteK16;
  } else if (
      kernel_name.rfind("sdpa_compute_", 0) == 0 ||
      kernel_name == "sdpa_softmax") {
    bits = kRoutePrefill | kRouteMaterializedAttention;
  } else if (kernel_name == "fd_split" || kernel_name == "fd_reduce") {
    bits = kRouteFlashDecoding;
  } else if (kernel_name == "linear_q4gsw_coop4_bicol_subgroup") {
    bits = kRouteBicolSubgroup;
  } else if (kernel_name.rfind("linear_q4gsw_bk64_qkv", 0) == 0) {
    bits = kRouteT1Bk64Qkv;
  } else if (kernel_name.rfind("linear_q4gsw_bk64", 0) == 0) {
    bits = kRouteT1Bk64;
  } else if (kernel_name.rfind("linear_q4gsw_paired_gate_up", 0) == 0) {
    bits = kRouteT2PairedGateUp;
  } else if (kernel_name == "silu_mul_fused") {
    bits = kRouteFusedSwiGlu;
  } else if (kernel_name.rfind("linear_q4gsw_steel", 0) == 0) {
    bits = kRouteT0Steel;
  } else if (kernel_name.rfind("linear_q4gsw", 0) == 0) {
    bits = kRouteGenericFallback;
  }

  constexpr uint32_t kAttentionRoutes = kRouteK16 | kRouteK16CausalBound |
      kRouteMaterializedAttention | kRouteFlashDecoding;
  const uint32_t new_attention = bits & kAttentionRoutes;
  const uint32_t prior_attention = g_last_route_mask & kAttentionRoutes;
  if (new_attention != 0 && prior_attention != 0 &&
      (new_attention & prior_attention) == 0) {
    ++g_last_route_conflict_count;
  }
  g_last_route_mask |= bits;
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

void WebGPUGraph::execute(const WebGPUGraphExecutionOptions& options) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  g_last_route_mask = 0;
  g_last_route_conflict_count = 0;
#endif // WGPU_BACKEND_ENABLE_PROFILING
  const size_t n = dispatches_.size();
  const size_t chunk = execute_config_.chunk_size;
  std::vector<bool> enabled_dispatches(n, true);
  for (size_t i = 0; i < n; i++) {
    if (dispatches_[i].kind != WebGPUDispatch::Kind::Compute) {
      continue;
    }
    const bool zero_x = dispatches_[i].workgroup_count_x == 0;
    const bool zero_y = dispatches_[i].workgroup_count_y == 0;
    if (zero_x != zero_y) {
      throw std::runtime_error("WebGPU: dispatch has a half-zero grid");
    }
    enabled_dispatches[i] = !zero_x;
  }
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      n,
      output_copies_.size(),
      execute_config_,
      suppressible_outputs_,
      options,
      enabled_dispatches);

  if (chunk == 0 || n <= chunk) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    size_t active_compute_count = 0;
    for (size_t i : plan.dispatch_chunks.front()) {
      if (dispatches_[i].kind == WebGPUDispatch::Kind::Compute) {
        active_compute_count++;
      }
    }
    // Bench: timestamp-query pool, null unless env-gated + feature present.
    WebGPUQueryPool* qp = nullptr;
    if (should_timestamp_query() && active_compute_count > 0) {
      if (auto* ctx = get_default_webgpu_context()) {
        if (ctx->timestamp_supported) {
          if (!ctx->querypool ||
              ctx->querypool->capacity() < active_compute_count) {
            ctx->querypool = std::make_unique<WebGPUQueryPool>();
            ctx->querypool->initialize(
                device_, static_cast<uint32_t>(active_compute_count));
          }
          qp = ctx->querypool.get();
          qp->reset(static_cast<uint32_t>(active_compute_count));
        }
      }
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING

    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    // One pass per dispatch: enforces storage RAW ordering across deps.
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    uint32_t query_index = 0;
#endif
    for (size_t i : plan.dispatch_chunks.front()) {
      const auto& dispatch = dispatches_[i];
      if (dispatch.kind == WebGPUDispatch::Kind::Copy) {
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            dispatch.copy_src,
            0,
            dispatch.copy_dst,
            0,
            dispatch.copy_nbytes);
        continue;
      }
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      record_active_route(dispatch.kernel_name);
#endif // WGPU_BACKEND_ENABLE_PROFILING
      WGPUComputePassDescriptor pass_desc = {};
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      // tw must outlive BeginComputePass (the descriptor points at it).
      WGPUPassTimestampWrites tw = {};
      if (qp) {
        tw = qp->writes_for(query_index);
        pass_desc.timestampWrites = &tw;
      }
#endif // WGPU_BACKEND_ENABLE_PROFILING
      WGPUComputePassEncoder pass =
          wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      wgpuComputePassEncoderSetPipeline(pass, dispatch.pipeline);
      wgpuComputePassEncoderSetBindGroup(
          pass, 0, dispatch.bind_group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(
          pass, dispatch.workgroup_count_x, dispatch.workgroup_count_y, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      if (qp) {
        qp->record(
            query_index,
            dispatch.kernel_name,
            {dispatch.workgroup_count_x, dispatch.workgroup_count_y, 1},
            {1, 1, 1});
        query_index++;
      }
#endif // WGPU_BACKEND_ENABLE_PROFILING
    }

    for (size_t i = 0; i < output_copies_.size(); i++) {
      if (!plan.copy_outputs[i]) {
        continue;
      }
      const auto& copy = output_copies_[i];
      wgpuCommandEncoderCopyBufferToBuffer(
          encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy.nbytes);
    }

#ifdef WGPU_BACKEND_ENABLE_PROFILING
    if (qp) {
      qp->resolve(encoder);
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

#ifdef WGPU_BACKEND_ENABLE_PROFILING
    if (qp) {
      qp->extract_results(instance_);
      qp->print_results();
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING
    return;
  }

  // GPU timestamp queries assume one submit; chunked execute is multi-submit.
  if (should_timestamp_query()) {
    throw std::runtime_error(
        "WebGPU: WEBGPU_TIMESTAMP_QUERY is incompatible with chunked execute "
        "(multi-submit); disable chunking to use GPU timestamp queries");
  }

  for (size_t chunk_index = 0; chunk_index < plan.dispatch_chunks.size();
       chunk_index++) {
    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    for (size_t i : plan.dispatch_chunks[chunk_index]) {
      if (dispatches_[i].kind == WebGPUDispatch::Kind::Copy) {
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            dispatches_[i].copy_src,
            0,
            dispatches_[i].copy_dst,
            0,
            dispatches_[i].copy_nbytes);
        continue;
      }
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      record_active_route(dispatches_[i].kernel_name);
#endif // WGPU_BACKEND_ENABLE_PROFILING
      WGPUComputePassDescriptor pass_desc = {};
      WGPUComputePassEncoder pass =
          wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      wgpuComputePassEncoderSetPipeline(pass, dispatches_[i].pipeline);
      wgpuComputePassEncoderSetBindGroup(
          pass, 0, dispatches_[i].bind_group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(
          pass,
          dispatches_[i].workgroup_count_x,
          dispatches_[i].workgroup_count_y,
          1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
    }

    if (chunk_index + 1 == plan.dispatch_chunks.size()) {
      for (size_t i = 0; i < output_copies_.size(); i++) {
        if (!plan.copy_outputs[i]) {
          continue;
        }
        const auto& copy = output_copies_[i];
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy.nbytes);
      }
    }

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
  }
}

namespace {

struct MapCallbackData {
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

void buffer_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  auto* data = static_cast<MapCallbackData*>(userdata1);
  data->status = status;
}

} // namespace

void WebGPUGraph::copy_outputs(
    std::vector<OutputData>& outputs,
    const WebGPUGraphExecutionOptions& options) {
  const size_t count = std::min(outputs.size(), output_staging_buffers_.size());
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      dispatches_.size(),
      output_copies_.size(),
      execute_config_,
      suppressible_outputs_,
      options);

  std::vector<MapCallbackData> cb_data(count);
  std::vector<WGPUFuture> map_futures(count, WGPUFuture{});

  // Single source of truth for the fp16-widen predicate + host map size, shared
  // by the map-request and map-result loops so the two cannot drift apart.
  auto output_map_size = [&](size_t i) -> std::pair<bool, size_t> {
    const auto& tensor = tensors_[output_ids_[i]];
    const bool widen_fp16 = !tensor.is_int && tensor.elem_size == 2 &&
        outputs[i].nbytes == tensor.cur_nbytes * 2;
    return {widen_fp16, widen_fp16 ? tensor.cur_nbytes : outputs[i].nbytes};
  };

  // Validate dtypes up front, before any wgpuBufferMapAsync is issued: require
  // an explicit fp32 host dtype for the fp16->fp32 widen (mirrors the
  // copy_inputs narrow guard) so a same-2:1-ratio non-fp32 host is not silently
  // reinterpreted as fp32. Throwing here — rather than mid map-request loop —
  // keeps in-flight async maps (whose callbacks point at cb_data) from being
  // left dangling when the exception unwinds this frame.
  for (size_t i = 0; i < count; i++) {
    if (!plan.copy_outputs[i] || outputs[i].nbytes == 0) {
      continue;
    }
    if (output_map_size(i).first && !outputs[i].host_is_fp32) {
      throw std::runtime_error(
          "WebGPU: fp16 device output requires an fp32 host tensor");
    }
  }

  for (size_t i = 0; i < count; i++) {
    if (!plan.copy_outputs[i] || outputs[i].nbytes == 0) {
      cb_data[i].status = WGPUMapAsyncStatus_Success;
      continue;
    }
    WGPUBufferMapCallbackInfo cb_info = {};
    cb_info.mode = WGPUCallbackMode_WaitAnyOnly;
    cb_info.callback = buffer_map_callback;
    cb_info.userdata1 = &cb_data[i];
    const size_t map_nbytes = output_map_size(i).second;
    map_futures[i] = wgpuBufferMapAsync(
        output_staging_buffers_[i], WGPUMapMode_Read, 0, map_nbytes, cb_info);
  }

  for (size_t i = 0; i < count; i++) {
    if (plan.copy_outputs[i] && outputs[i].nbytes != 0 &&
        webgpu_wait(instance_, map_futures[i]) != WGPUWaitStatus_Success) {
      throw std::runtime_error("WebGPU: WaitAny failed for output map");
    }
  }

  for (size_t i = 0; i < count; i++) {
    if (!plan.copy_outputs[i] || outputs[i].nbytes == 0) {
      continue;
    }
    if (cb_data[i].status == WGPUMapAsyncStatus_Success) {
      const auto [widen_fp16, map_nbytes] = output_map_size(i);
      const void* mapped = wgpuBufferGetConstMappedRange(
          output_staging_buffers_[i], 0, map_nbytes);
      if (widen_fp16) {
        const auto* src =
            static_cast<const executorch::runtime::etensor::Half*>(mapped);
        auto* dst = static_cast<float*>(outputs[i].data);
        const size_t numel = map_nbytes / sizeof(*src);
        for (size_t e = 0; e < numel; e++) {
          dst[e] = static_cast<float>(src[e]);
        }
      } else {
        std::memcpy(outputs[i].data, mapped, outputs[i].nbytes);
      }
      wgpuBufferUnmap(output_staging_buffers_[i]);
    } else {
      throw std::runtime_error("WebGPU buffer map failed for output");
    }
  }
}

WebGPUMemoryStats WebGPUGraph::memory_stats() const {
  WebGPUMemoryStats stats;
  for (size_t i = 0; i < value_types_.size(); i++) {
    if (value_types_[i] == ValueType::Tensor && tensors_[i].nbytes > 0) {
      stats.num_tensors++;
      // Shared tensors are tracked via shared_buffer_sizes_; a deferred
      // prepack-routed constant has no buffer (no GPU memory) -> not counted.
      bool is_shared =
          i < tensor_mem_obj_ids_.size() && tensor_mem_obj_ids_[i] >= 0;
      if (!is_shared && tensors_[i].buffer != nullptr) {
        stats.unshared_tensor_buffer_bytes += tensors_[i].nbytes;
      }
    }
  }
  for (size_t s : shared_buffer_sizes_) {
    stats.shared_buffer_bytes += s;
  }
  stats.num_shared_objects = static_cast<int>(shared_buffers_.size());
  stats.tensor_buffer_bytes =
      stats.shared_buffer_bytes + stats.unshared_tensor_buffer_bytes;
  for (size_t i = 0; i < output_ids_.size(); i++) {
    stats.staging_buffer_bytes += tensors_[output_ids_[i]].nbytes;
  }
  stats.uniform_buffer_bytes = uniform_buffer_bytes_;
  stats.num_dispatches = static_cast<int>(dispatches_.size());
  stats.num_cached_pipelines = static_cast<int>(pipeline_cache_.size());
  stats.num_cached_shaders = static_cast<int>(shader_cache_.size());
  return stats;
}

} // namespace executorch::backends::webgpu
