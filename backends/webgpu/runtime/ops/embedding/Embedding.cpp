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
#include <executorch/backends/webgpu/runtime/ops/embedding/embedding_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct EmbeddingParams {
  uint32_t embed_dim;
  uint32_t num_elements;
  uint32_t _pad[2];
};
static_assert(
    sizeof(EmbeddingParams) == 16,
    "EmbeddingParams must be 16 bytes");

// aten.embedding.default args: [weight, indices, padding_idx,
// scale_grad_by_freq, sparse, out]. Forward is a plain row gather
// (padding_idx/scale_grad/sparse only affect the backward pass); out[row, :] =
// weight[indices[row], :]. fp32 weight, int32 indices (the backend's index
// convention; mirrors index.wgsl / embedding_q4gsw).
void embedding_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int weight_id = args.at(0);
  const int indices_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& weight = graph.get_tensor(weight_id);
  const auto& indices = graph.get_tensor(indices_id);
  const auto& out = graph.get_tensor(out_id);

  if (weight.buffer == nullptr || indices.buffer == nullptr ||
      out.buffer == nullptr) {
    throw std::runtime_error("WebGPU embedding: null buffer binding");
  }
  if (weight.dims.size() < 2 || out.dims.empty() || indices.dims.empty()) {
    throw std::runtime_error("WebGPU embedding: malformed dims");
  }

  const uint32_t embed_dim = static_cast<uint32_t>(out.dims.back());
  if (embed_dim == 0) {
    throw std::runtime_error("WebGPU embedding: zero embed_dim");
  }
  if (static_cast<uint32_t>(weight.dims.back()) != embed_dim) {
    throw std::runtime_error(
        "WebGPU embedding: weight row width != out embed_dim");
  }

  const uint64_t out_numel = utils::numel(out.dims);
  const uint32_t num_indices = static_cast<uint32_t>(out_numel / embed_dim);

  const uint64_t indices_numel = utils::numel(indices.dims);
  // Per-type byte guards (no runtime dtype): indices int32, weight/out fp32.
  if (indices_numel != num_indices ||
      indices.nbytes != indices_numel * sizeof(int32_t)) {
    throw std::runtime_error(
        "WebGPU embedding: dtype/byte-size mismatch (indices int32, out fp32)");
  }
  utils::check_fp32(out, "embedding", "output");

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kEmbeddingWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "embedding");

  EmbeddingParams params = {};
  params.embed_dim = embed_dim;
  params.num_elements = static_cast<uint32_t>(out_numel);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(EmbeddingParams));
  graph.add_uniform_buffer_bytes(sizeof(EmbeddingParams));

  WGPUConstantEntry wg_size_constant = utils::make_wg_size_constant(wg_size);

  // out (rw) + indices/weight (ro storage) + uniform.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kEmbeddingWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           indices.buffer,
           indices.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight.buffer,
           weight.nbytes},
          {3,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(EmbeddingParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.embedding.default, embedding_impl);
}

} // namespace executorch::backends::webgpu
