/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/embedding_q4gsw/embedding_q4gsw_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned, 48 bytes).
struct EmbeddingParams {
  uint32_t embed_dim;
  uint32_t blocks_per_row;
  uint32_t num_indices;
  uint32_t group_size;
  uint32_t groups_per_row;
  uint32_t bytes_per_row;
  uint32_t total_blocks;
  uint32_t is_linear_weight;
  uint32_t row_lo;
  uint32_t rows_in_chunk;
  uint32_t pad0;
  uint32_t pad1;
};
static_assert(
    sizeof(EmbeddingParams) == 48,
    "EmbeddingParams must be 48 bytes");

struct EmbeddingLayout {
  uint32_t embed_dim;
  uint32_t blocks_per_row;
  uint32_t group_size;
  uint32_t groups_per_row;
  uint32_t bytes_per_row;
  bool is_linear_weight;
};

struct EmbeddingChunkSpec {
  uint32_t row_lo;
  uint32_t rows;
  uint64_t weight_offset;
  uint64_t weight_size;
  uint64_t scales_offset;
  uint64_t scales_size;
};

struct EmbeddingChunkRuntime {
  EmbeddingChunkSpec spec;
  size_t dispatch_index;
  WGPUBuffer params_buffer;
};

EmbeddingParams make_embedding_params(
    const EmbeddingLayout& layout,
    uint32_t num_indices,
    uint32_t total_blocks,
    uint32_t row_lo,
    uint32_t rows_in_chunk) {
  return {
      layout.embed_dim,
      layout.blocks_per_row,
      num_indices,
      layout.group_size,
      layout.groups_per_row,
      layout.bytes_per_row,
      total_blocks,
      layout.is_linear_weight ? 1u : 0u,
      row_lo,
      rows_in_chunk,
      0u,
      0u};
}

uint64_t checked_lcm(uint64_t a, uint64_t b) {
  const uint64_t divisor = std::gcd(a, b);
  if (a > std::numeric_limits<uint64_t>::max() / (b / divisor)) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: row-alignment quantum overflows");
  }
  return a * (b / divisor);
}

std::vector<EmbeddingChunkSpec> make_embedding_chunks(
    uint64_t max_binding_bytes,
    uint64_t min_offset_alignment,
    uint64_t max_buffer_bytes,
    uint32_t vocab_rows,
    uint64_t weight_bytes_per_row,
    uint64_t scales_bytes_per_row,
    uint64_t weight_buffer_bytes,
    uint64_t scales_buffer_bytes) {
  if (max_binding_bytes == 0u || min_offset_alignment == 0u ||
      max_buffer_bytes == 0u || vocab_rows == 0u ||
      weight_bytes_per_row == 0u || scales_bytes_per_row == 0u) {
    throw std::runtime_error("WebGPU embedding_q4gsw: invalid chunking limits");
  }
  if (weight_buffer_bytes > max_buffer_bytes ||
      scales_buffer_bytes > max_buffer_bytes) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: tensor exceeds maxBufferSize");
  }
  if (weight_buffer_bytes <= max_binding_bytes &&
      scales_buffer_bytes <= max_binding_bytes) {
    return {{0u, vocab_rows, 0u, weight_buffer_bytes, 0u, scales_buffer_bytes}};
  }

  const uint64_t weight_row_quantum = min_offset_alignment /
      std::gcd(min_offset_alignment, weight_bytes_per_row);
  const uint64_t scales_row_quantum = min_offset_alignment /
      std::gcd(min_offset_alignment, scales_bytes_per_row);
  const uint64_t row_quantum =
      checked_lcm(weight_row_quantum, scales_row_quantum);
  const uint64_t max_rows = std::min(
      max_binding_bytes / weight_bytes_per_row,
      max_binding_bytes / scales_bytes_per_row);
  const uint64_t rows_per_chunk = (max_rows / row_quantum) * row_quantum;
  if (rows_per_chunk == 0u) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: no aligned row chunk fits binding limit");
  }

  std::vector<EmbeddingChunkSpec> chunks;
  for (uint64_t row_lo = 0u; row_lo < vocab_rows; row_lo += rows_per_chunk) {
    const uint64_t rows =
        std::min<uint64_t>(rows_per_chunk, vocab_rows - row_lo);
    const uint64_t weight_offset = row_lo * weight_bytes_per_row;
    const uint64_t scales_offset = row_lo * scales_bytes_per_row;
    const uint64_t weight_size = rows * weight_bytes_per_row;
    const uint64_t scales_size = rows * scales_bytes_per_row;
    if (weight_offset % min_offset_alignment != 0u ||
        scales_offset % min_offset_alignment != 0u ||
        weight_size > max_binding_bytes || scales_size > max_binding_bytes ||
        weight_offset > weight_buffer_bytes ||
        weight_size > weight_buffer_bytes - weight_offset ||
        scales_offset > scales_buffer_bytes ||
        scales_size > scales_buffer_bytes - scales_offset ||
        row_lo > UINT32_MAX || rows > UINT32_MAX) {
      throw std::runtime_error(
          "WebGPU embedding_q4gsw: invalid aligned chunk binding");
    }
    chunks.push_back(
        {static_cast<uint32_t>(row_lo),
         static_cast<uint32_t>(rows),
         weight_offset,
         weight_size,
         scales_offset,
         scales_size});
  }
  return chunks;
}

// Resize hook body: recompute counts/dispatch; out = indices dims +
// [embed_dim].
void resize_embedding_q4gsw(
    WebGPUGraph& g,
    int indices_id,
    int out_id,
    const EmbeddingLayout& layout,
    uint32_t wg_size,
    const std::vector<EmbeddingChunkRuntime>& chunks) {
  const auto& id = g.cur_dims(indices_id);
  const uint64_t ni = utils::numel_of(id);
  if (ni == 0) {
    throw std::runtime_error("WebGPU embedding_q4gsw: zero indices");
  }
  const uint64_t total_blocks = ni * layout.blocks_per_row;
  if (total_blocks > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: total_blocks exceeds uint32");
  }
  std::vector<int64_t> od = id;
  od.push_back(static_cast<int64_t>(layout.embed_dim));
  g.set_cur_dims(out_id, od);
  const utils::WgCount grid = utils::compute_2d_workgroup_count(
      g.device(),
      static_cast<uint32_t>(total_blocks),
      wg_size,
      "embedding_q4gsw(resize)");
  for (const EmbeddingChunkRuntime& chunk : chunks) {
    const EmbeddingParams p = make_embedding_params(
        layout,
        static_cast<uint32_t>(ni),
        static_cast<uint32_t>(total_blocks),
        chunk.spec.row_lo,
        chunk.spec.rows);
    wgpuQueueWriteBuffer(g.queue(), chunk.params_buffer, 0, &p, sizeof(p));
    WebGPUDispatch& dispatch = g.dispatch_at(chunk.dispatch_index);
    dispatch.workgroup_count_x = grid.x;
    dispatch.workgroup_count_y = grid.y;
  }
}

// arg order mirrors Vulkan EmbeddingQ4gsw.cpp.
void embedding_q4gsw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int weight_id = args.at(0);
  const int scales_id = args.at(1);
  const int group_size_id = args.at(2);
  const int indices_id = args.at(3);
  const int is_linear_weight_id = args.at(4);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();

  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& indices = graph.get_tensor(indices_id);
  const auto& out = graph.get_tensor(out_id);

  // is_linear_weight selects the nibble packing (false: even dim = high nibble;
  // true: even dim = low nibble). The shader handles both via a uniform.
  bool is_linear = false;
  if (graph.get_value_type(is_linear_weight_id) ==
      WebGPUGraph::ValueType::Bool) {
    is_linear = graph.get_bool(is_linear_weight_id);
  } else if (
      graph.get_value_type(is_linear_weight_id) ==
      WebGPUGraph::ValueType::Int) {
    is_linear = graph.get_int(is_linear_weight_id) != 0;
  } else {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: is_linear_weight must be Bool or Int");
  }

  if (weight.dims.size() < 2 || scales.dims.size() < 2 || out.dims.empty() ||
      indices.dims.empty()) {
    throw std::runtime_error("WebGPU embedding_q4gsw: malformed dims");
  }

  if (out.dims.back() <= 0 ||
      static_cast<uint64_t>(out.dims.back()) > UINT32_MAX ||
      weight.dims[0] <= 0 ||
      static_cast<uint64_t>(weight.dims[0]) > UINT32_MAX ||
      weight.dims[1] <= 0 || scales.dims[0] <= 0 || scales.dims[1] <= 0 ||
      static_cast<uint64_t>(scales.dims[1]) > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: dimensions exceed supported range");
  }
  const uint32_t embed_dim = static_cast<uint32_t>(out.dims.back());
  const uint32_t vocab_rows = static_cast<uint32_t>(weight.dims[0]);
  if (embed_dim % 32u != 0u) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: embed_dim must be a nonzero multiple of 32");
  }
  if (static_cast<uint64_t>(weight.dims[1]) * 2 != embed_dim) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: weight row stride mismatch (embed_dim/2)");
  }

  int64_t group_size = 0;
  if (graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int) {
    group_size = graph.get_int(group_size_id);
  }
  if (group_size <= 0 || static_cast<uint64_t>(group_size) > UINT32_MAX) {
    throw std::runtime_error("WebGPU embedding_q4gsw: group_size out of range");
  }

  // Leading index dims flatten row-major (mirrors Vulkan num_indices).
  const uint64_t out_numel = utils::numel_of(out.dims);
  if (out_numel % embed_dim != 0u || out_numel / embed_dim == 0u ||
      out_numel / embed_dim > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: invalid number of indices");
  }
  const uint32_t num_indices = static_cast<uint32_t>(out_numel / embed_dim);
  const uint32_t groups_per_row = static_cast<uint32_t>(scales.dims[1]);
  const uint32_t blocks_per_row = embed_dim / 32u;
  const uint32_t bytes_per_row = embed_dim / 2u;
  const uint64_t total_blocks =
      static_cast<uint64_t>(num_indices) * blocks_per_row;
  if (static_cast<uint64_t>(groups_per_row) * group_size != embed_dim) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: groups_per_row * group_size != embed_dim");
  }
  if (scales.dims[0] != weight.dims[0]) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: weight/scales vocab rows differ");
  }
  if (weight.buffer == nullptr || scales.buffer == nullptr ||
      indices.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("WebGPU embedding_q4gsw: null buffer binding");
  }

  // Per-type byte guards (no runtime dtype): indices i32, weight u8, fp32 rest.
  const uint64_t indices_numel = utils::numel_of(indices.dims);
  const uint64_t weight_numel = utils::numel_of(weight.dims);
  const uint64_t scales_numel = utils::numel_of(scales.dims);
  if (indices_numel != num_indices || !indices.is_int ||
      indices.elem_size != sizeof(int32_t) ||
      indices.nbytes != indices_numel * sizeof(int32_t) ||
      weight.nbytes != weight_numel ||
      weight_numel != static_cast<uint64_t>(vocab_rows) * bytes_per_row ||
      scales_numel != static_cast<uint64_t>(vocab_rows) * groups_per_row ||
      !utils::is_fp32_tensor(scales) || !utils::is_fp32_tensor(out)) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: dtype/byte-size mismatch "
        "(indices int32, weight uint8, scales/out fp32)");
  }
  if (total_blocks > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: total_blocks exceeds uint32 dispatch range");
  }

  std::vector<int64_t> expected_out_dims = indices.dims;
  expected_out_dims.push_back(embed_dim);
  if (out.dims != expected_out_dims) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: output shape must be indices + embed_dim");
  }

  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success ||
      limits.maxStorageBufferBindingSize == 0u || limits.maxBufferSize == 0u ||
      limits.minStorageBufferOffsetAlignment == 0u) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: failed to query binding limits");
  }
  if (indices.nbytes > limits.maxStorageBufferBindingSize ||
      out.nbytes > limits.maxStorageBufferBindingSize ||
      indices.nbytes > limits.maxBufferSize ||
      out.nbytes > limits.maxBufferSize) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: indices/output exceed binding limits");
  }
  const std::vector<EmbeddingChunkSpec> chunk_specs = make_embedding_chunks(
      limits.maxStorageBufferBindingSize,
      limits.minStorageBufferOffsetAlignment,
      limits.maxBufferSize,
      vocab_rows,
      bytes_per_row,
      static_cast<uint64_t>(groups_per_row) * sizeof(float),
      weight.nbytes,
      scales.nbytes);

  // 1D dispatch: one thread per 32-dim block; validate before any alloc.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kEmbeddingQ4gswWorkgroupSizeX);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(total_blocks), wg_size, "embedding_q4gsw");

  const EmbeddingLayout layout = {
      embed_dim,
      blocks_per_row,
      static_cast<uint32_t>(group_size),
      groups_per_row,
      bytes_per_row,
      is_linear};
  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  std::vector<EmbeddingChunkRuntime> chunks;
  chunks.reserve(chunk_specs.size());
  for (const EmbeddingChunkSpec& chunk : chunk_specs) {
    const EmbeddingParams params = make_embedding_params(
        layout,
        num_indices,
        static_cast<uint32_t>(total_blocks),
        chunk.row_lo,
        chunk.rows);
    WGPUBuffer params_buffer = graph.create_params_buffer(params);
    graph.add_uniform_buffer_bytes(sizeof(params));

    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kEmbeddingQ4gswWGSL,
        {
            {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
            {1,
             WGPUBufferBindingType_ReadOnlyStorage,
             indices.buffer,
             indices.nbytes},
            {2,
             WGPUBufferBindingType_ReadOnlyStorage,
             weight.buffer,
             chunk.weight_size,
             chunk.weight_offset},
            {3,
             WGPUBufferBindingType_ReadOnlyStorage,
             scales.buffer,
             chunk.scales_size,
             chunk.scales_offset},
            {4,
             WGPUBufferBindingType_Uniform,
             params_buffer,
             sizeof(EmbeddingParams)},
        },
        &wg_size_constant,
        1);

    const size_t dispatch_index = graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         workgroup_count.x,
         "embedding_q4gsw",
         workgroup_count.y});
    chunks.push_back({chunk, dispatch_index, params_buffer});
  }

  // Dynamic shapes: recompute counts/dispatch; out = indices + [embed_dim].
  graph.add_tensor_resize_hook(
      indices_id,
      [indices_id, out_id, layout, wg_size, chunks](WebGPUGraph& g) {
        resize_embedding_q4gsw(g, indices_id, out_id, layout, wg_size, chunks);
      });
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.embedding_q4gsw.default, embedding_q4gsw_impl);
}

} // namespace executorch::backends::webgpu
