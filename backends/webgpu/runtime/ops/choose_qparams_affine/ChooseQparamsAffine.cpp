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
#include <memory>
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

struct ChooseQParamsState {
  ChooseQParamsParams params;
  std::vector<int64_t> output_dims;
  utils::WgCount grid;
};

ChooseQParamsState make_choose_qparams_state(
    WGPUDevice device,
    const std::vector<int64_t>& input_dims,
    uint32_t max_rows,
    uint32_t reduce_size,
    int32_t quant_min,
    int32_t quant_max) {
  if (input_dims.empty() || input_dims.back() != reduce_size) {
    throw std::runtime_error(
        "choose_qparams_affine: live reduce size mismatch");
  }
  const uint64_t numel = utils::numel_of(input_dims);
  if (numel == 0u || numel % reduce_size != 0u) {
    throw std::runtime_error("choose_qparams_affine: invalid live input numel");
  }
  const uint64_t rows = numel / reduce_size;
  if (rows == 0u || rows > max_rows || rows > UINT32_MAX) {
    throw std::runtime_error(
        "choose_qparams_affine: live rows exceed the build-time max");
  }

  ChooseQParamsState state = {};
  state.params = {
      static_cast<uint32_t>(rows), reduce_size, quant_min, quant_max};
  state.output_dims = input_dims;
  state.output_dims.pop_back();
  state.grid = utils::compute_2d_workgroup_count(
      device,
      utils::div_up(static_cast<uint32_t>(rows), 4u),
      1u,
      "choose_qparams_affine");
  return state;
}

// torchao.choose_qparams_affine args (mirrors Vulkan ChooseQParams.cpp:158):
// [input, mapping_type, block_size, target_dtype, quant_min, quant_max, eps,
//  scale_dtype, zero_point_dtype, keepdim, out_tuple(scale, zp)].
// Routes to the per-row (last-dim) path.
void choose_qparams_affine_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  if (args.size() != 11u) {
    throw std::runtime_error(
        "choose_qparams_affine: expected 10 inputs plus output");
  }
  const int in_id = args.at(0);
  const int mapping_type_id = args.at(1);
  const int block_size_id = args.at(2);
  const int target_dtype_id = args.at(3);
  const int quant_min_id = args.at(4);
  const int quant_max_id = args.at(5);
  const int eps_id = args.at(6);
  const int scale_dtype_id = args.at(7);
  const int zero_point_dtype_id = args.at(8);
  const int keepdim_id = args.at(9);
  const int out_list_id = args.at(10);

  using VT = WebGPUGraph::ValueType;
  if (graph.get_value_type(mapping_type_id) != VT::String ||
      graph.get_string(mapping_type_id) != "ASYMMETRIC") {
    throw std::runtime_error(
        "choose_qparams_affine: only ASYMMETRIC mapping is supported");
  }
  if (graph.get_value_type(block_size_id) != VT::IntList ||
      graph.get_value_type(quant_min_id) != VT::Int ||
      graph.get_value_type(quant_max_id) != VT::Int ||
      graph.get_value_type(keepdim_id) != VT::Bool ||
      graph.get_value_type(out_list_id) != VT::ValueList) {
    throw std::runtime_error("choose_qparams_affine: malformed scalar args");
  }
  // The current Vulkan serializer encodes torch.dtype values as Null and
  // materializes the schema-default keepdim=false. The exact supported dtypes
  // are therefore validated from the output tensors below.
  if (graph.get_value_type(target_dtype_id) != VT::Null ||
      graph.get_value_type(eps_id) != VT::Null ||
      graph.get_value_type(scale_dtype_id) != VT::Null ||
      graph.get_value_type(zero_point_dtype_id) != VT::Null) {
    throw std::runtime_error(
        "choose_qparams_affine: unsupported serialized dtype arguments");
  }
  if (graph.get_bool(keepdim_id)) {
    throw std::runtime_error("choose_qparams_affine: keepdim must be false");
  }

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
  if (in.dims.empty() || in.is_int || in.elem_size != sizeof(float) ||
      scale_t.is_int || scale_t.elem_size != sizeof(float)) {
    throw std::runtime_error(
        "choose_qparams_affine: input and scale must be fp32");
  }

  if (in.dims.back() <= 0 ||
      static_cast<uint64_t>(in.dims.back()) > UINT32_MAX) {
    throw std::runtime_error("choose_qparams_affine: invalid last dimension");
  }
  const uint64_t reduce_size = static_cast<uint64_t>(in.dims.back());
  const uint64_t in_numel = utils::numel_of(in.dims);
  if (in_numel > SIZE_MAX / sizeof(float)) {
    throw std::runtime_error("choose_qparams_affine: input size overflows");
  }
  const uint64_t num_rows = in_numel / reduce_size;
  if (in_numel % reduce_size != 0u || num_rows == 0 || num_rows > UINT32_MAX) {
    throw std::runtime_error("choose_qparams_affine: bad row/reduce shape");
  }
  if (in.nbytes != static_cast<size_t>(in_numel) * sizeof(float)) {
    throw std::runtime_error("choose_qparams_affine: input must be fp32");
  }
  // scale is fp32[num_rows]; zp is int8[num_rows] (bound as array<u32>).
  if (scale_t.dims != zp_t.dims || utils::numel_of(scale_t.dims) != num_rows ||
      scale_t.nbytes != num_rows * sizeof(float)) {
    throw std::runtime_error("choose_qparams_affine: scale must be fp32[rows]");
  }
  // zp is int8[rows] (elem_size 1), packed 4-per-u32 in the shader. Buffers are
  // allocated max(align4(nbytes), 4), so a ragged tail block's whole-word store
  // lands in the pad, in-bounds; the shader clamps its row loop to num_rows.
  if (!zp_t.is_int8 || zp_t.elem_size != sizeof(int8_t) ||
      zp_t.nbytes != num_rows) {
    throw std::runtime_error("choose_qparams_affine: zp must be int8[rows]");
  }

  // The kernel implements only the asymmetric, per-row (last-dim), int8 path;
  // validate the schema args it assumes and fail loud rather than silently
  // ignoring them (mirrors Vulkan, which consumes block_size + quant_min/max).
  const int64_t quant_min = graph.get_int(quant_min_id);
  const int64_t quant_max = graph.get_int(quant_max_id);
  if (quant_min != -128 || quant_max != 127) {
    throw std::runtime_error(
        "choose_qparams_affine: only the int8 range [-128, 127] is supported");
  }
  // Per-row fast path: block_size must be [1, ..., 1, reduce_size].
  const std::vector<int64_t>& block_size = graph.get_int_list(block_size_id);
  if (block_size.size() != in.dims.size() ||
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

  const uint32_t max_rows = static_cast<uint32_t>(num_rows);
  const uint32_t reduce_size_u32 = static_cast<uint32_t>(reduce_size);
  const ChooseQParamsState initial_state = make_choose_qparams_state(
      device,
      in.dims,
      max_rows,
      reduce_size_u32,
      static_cast<int32_t>(quant_min),
      static_cast<int32_t>(quant_max));

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kChooseQparamsAffineWorkgroupSizeX);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf = utils::make_uniform(
      device, &initial_state.params, sizeof(ChooseQParamsParams));
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

  const size_t dispatch_index = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       initial_state.grid.x,
       "choose_qparams_affine",
       initial_state.grid.y});

  auto producer_elided = std::make_shared<bool>(false);
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       scale_id,
       zp_id,
       max_rows,
       reduce_size_u32,
       quant_min = static_cast<int32_t>(quant_min),
       quant_max = static_cast<int32_t>(quant_max),
       dispatch_index,
       producer_elided,
       params_buf](WebGPUGraph& g) {
        const ChooseQParamsState state = make_choose_qparams_state(
            g.device(),
            g.cur_dims(in_id),
            max_rows,
            reduce_size_u32,
            quant_min,
            quant_max);
        wgpuQueueWriteBuffer(
            g.queue(), params_buf, 0, &state.params, sizeof(state.params));
        auto& dispatch = g.dispatch_at(dispatch_index);
        dispatch.workgroup_count_x =
            utils::cqp_resize_workgroups(*producer_elided, state.grid.x);
        dispatch.workgroup_count_y =
            utils::cqp_resize_workgroups(*producer_elided, state.grid.y);
        g.set_cur_dims(scale_id, state.output_dims);
        g.set_cur_dims(zp_id, state.output_dims);
      });

  WebGPUGraph::CqpFusionSite site = {};
  site.input_id = in_id;
  site.scales_id = scale_id;
  site.zero_points_id = zp_id;
  site.rows = max_rows;
  site.row_width = reduce_size_u32;
  site.quant_min = quant_min;
  site.quant_max = quant_max;
  site.dispatch_index = dispatch_index;
  site.input_buffer = in.buffer;
  site.scales_buffer = scale_t.buffer;
  site.zero_points_buffer = zp_t.buffer;
  site.producer_elided = std::move(producer_elided);
  graph.offer_cqp_fusion_site(std::move(site));

  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      torchao.choose_qparams_affine.default, choose_qparams_affine_impl);
}

} // namespace executorch::backends::webgpu
