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
#include <executorch/backends/webgpu/runtime/ops/choose_qparams_affine/choose_qparams_affine_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct ChooseQParamsParams {
  uint32_t num_rows;
  uint32_t reduce_size;
  int32_t quant_min;
  int32_t quant_max;
};
static_assert(
    sizeof(ChooseQParamsParams) == 16,
    "ChooseQParamsParams must match the WGSL Params struct (16 bytes)");

// torchao.choose_qparams_affine args (mirrors Vulkan ChooseQParams.cpp:158):
// [input, mapping_type, block_size, target_dtype, quant_min, quant_max, eps,
//  scale_dtype, zero_point_dtype, out_tuple(scale, zp)]. Routes to the per-row
// (last-dim) path: one workgroup per row computes asymmetric scale/zp.
void choose_qparams_affine_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int out_list_id = args.at(args.size() - 1);

  const std::vector<int>& out_ids = graph.get_value_list(out_list_id);
  if (out_ids.size() != 2) {
    throw std::runtime_error(
        "choose_qparams_affine: expected 2 outputs (scale, zp)");
  }
  const int scale_id = out_ids.at(0);
  const int zp_id = out_ids.at(1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(scale_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(zp_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("choose_qparams_affine: in/scale/zp not tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& scale_t = graph.get_tensor(scale_id);
  const auto& zp_t = graph.get_tensor(zp_id);
  if (in.buffer == nullptr || scale_t.buffer == nullptr ||
      zp_t.buffer == nullptr) {
    throw std::runtime_error("choose_qparams_affine: null buffer binding");
  }
  if (in.dims.empty()) {
    throw std::runtime_error("choose_qparams_affine: input has no dims");
  }

  const uint64_t reduce_size = static_cast<uint64_t>(in.dims.back());
  if (reduce_size == 0) {
    throw std::runtime_error("choose_qparams_affine: last dim == 0");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint64_t num_rows = in_numel / reduce_size;
  if (num_rows == 0 || num_rows > UINT32_MAX || reduce_size > UINT32_MAX) {
    throw std::runtime_error("choose_qparams_affine: bad row/reduce shape");
  }
  if (in.nbytes != in_numel * sizeof(float)) {
    throw std::runtime_error("choose_qparams_affine: input must be fp32");
  }
  // scale is fp32[num_rows]; zp is int8[num_rows] (bound as array<u32>).
  if (scale_t.nbytes != num_rows * sizeof(float)) {
    throw std::runtime_error("choose_qparams_affine: scale must be fp32[rows]");
  }
  // zp is int8[rows] (elem_size 1), packed 4-per-u32 in the shader. int8
  // buffers are allocated max(nbytes, 4); M<=4 pads to one word, M%4==0 is
  // word-exact. Other M (5,6,7,...) would overflow the M-byte buffer -> reject.
  if (!zp_t.is_int8 || zp_t.nbytes != num_rows) {
    throw std::runtime_error("choose_qparams_affine: zp must be int8[rows]");
  }
  if (num_rows > 4 && num_rows % 4 != 0) {
    throw std::runtime_error(
        "choose_qparams_affine: num_rows must be <=4 or a multiple of 4");
  }

  // The kernel implements only the asymmetric, per-row (last-dim), int8 path;
  // validate the schema args it assumes and fail loud rather than silently
  // ignoring them (mirrors Vulkan, which consumes block_size + quant_min/max).
  const int quant_min_id = args.at(4);
  const int quant_max_id = args.at(5);
  if (graph.get_value_type(quant_min_id) != WebGPUGraph::ValueType::Int ||
      graph.get_value_type(quant_max_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error(
        "choose_qparams_affine: quant_min/quant_max must be int scalars");
  }
  const int64_t quant_min = graph.get_int(quant_min_id);
  const int64_t quant_max = graph.get_int(quant_max_id);
  if (quant_min != -128 || quant_max != 127) {
    throw std::runtime_error(
        "choose_qparams_affine: only the int8 range [-128, 127] is supported");
  }
  // Per-row fast path: block_size must be [1, ..., 1, reduce_size].
  const int block_size_id = args.at(2);
  if (graph.get_value_type(block_size_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error(
        "choose_qparams_affine: block_size must be an int list");
  }
  const std::vector<int64_t>& block_size = graph.get_int_list(block_size_id);
  if (block_size.empty() ||
      block_size.back() != static_cast<int64_t>(reduce_size)) {
    throw std::runtime_error(
        "choose_qparams_affine: block_size must reduce the last dim");
  }
  for (size_t d = 0; d + 1 < block_size.size(); ++d) {
    if (block_size[d] != 1) {
      throw std::runtime_error(
          "choose_qparams_affine: only per-row (last-dim) blocks are supported");
    }
  }

  ChooseQParamsParams params = {};
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.reduce_size = static_cast<uint32_t>(reduce_size);
  params.quant_min = static_cast<int32_t>(quant_min);
  params.quant_max = static_cast<int32_t>(quant_max);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kChooseQparamsAffineWorkgroupSizeX);
  // One workgroup per block of 4 rows (wg_size threads cooperate per row); the
  // block packs its 4 int8 zps into one u32. 2D-fold lifts the 65535 grid cap.
  const uint32_t num_blocks = static_cast<uint32_t>((num_rows + 3) / 4);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_blocks, 1, "choose_qparams_affine");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(ChooseQParamsParams));
  graph.add_uniform_buffer_bytes(sizeof(ChooseQParamsParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kChooseQparamsAffineWGSL,
      {
          {0, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {1, WGPUBufferBindingType_Storage, scale_t.buffer, scale_t.nbytes},
          {2,
           WGPUBufferBindingType_Storage,
           zp_t.buffer,
           // Bind word-aligned (buffer is >= max(nbytes,4); array<u32> needs a
           // mult of 4).
           ((zp_t.nbytes + 3u) / 4u) * 4u},
          {3,
           WGPUBufferBindingType_Uniform,
           params_buf,
           sizeof(ChooseQParamsParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "choose_qparams_affine",
       workgroup_count.y});

  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      torchao.choose_qparams_affine.default, choose_qparams_affine_impl);
}

} // namespace executorch::backends::webgpu
