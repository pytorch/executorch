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
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_vec4_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct ConvParams {
  uint32_t B;
  uint32_t IC;
  uint32_t IH;
  uint32_t IW;
  uint32_t OC;
  uint32_t OH;
  uint32_t OW;
  uint32_t KH;
  uint32_t KW;
  uint32_t sH;
  uint32_t sW;
  uint32_t pH;
  uint32_t pW;
  uint32_t dH;
  uint32_t dW;
  uint32_t groups;
  uint32_t has_bias;
  uint32_t _p0;
  uint32_t _p1;
  uint32_t _p2;
};
static_assert(
    sizeof(ConvParams) == 80,
    "ConvParams must be 80 bytes (16-mult)");

// Transposed 2D convolution (gather form), folded into the convolution handler
// (C2: a 2nd WEBGPU_REGISTER_OP(aten.convolution.default) would be silently
// dropped). weight layout = torch convT [IC, OC/groups, KH, KW]. CPU-derisked
// vs torch.conv_transpose2d to fp64 round-off incl. non-square spatial +
// kernel.
void conv_transpose2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int stride_id = args.at(3);
  const int padding_id = args.at(4);
  const int dilation_id = args.at(5);
  const int output_padding_id = args.at(7);
  const int groups_id = args.at(8);
  const int out_id = args.at(9);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& out = graph.get_tensor(out_id);
  if (in.dims.size() != 4 || weight.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error(
        "WebGPU conv_transpose2d: expected 4D in/weight/out");
  }

  uint32_t sH, sW, pH, pW, dH, dW, opH, opW;
  utils::parse_hw(
      graph.get_int_list(stride_id), sH, sW, "conv_transpose2d", "stride");
  utils::parse_hw(
      graph.get_int_list(padding_id), pH, pW, "conv_transpose2d", "padding");
  utils::parse_hw(
      graph.get_int_list(dilation_id), dH, dW, "conv_transpose2d", "dilation");
  utils::parse_hw(
      graph.get_int_list(output_padding_id),
      opH,
      opW,
      "conv_transpose2d",
      "output_padding");

  const uint32_t B = static_cast<uint32_t>(in.dims[0]);
  const uint32_t IC = static_cast<uint32_t>(in.dims[1]);
  const uint32_t IH = static_cast<uint32_t>(in.dims[2]);
  const uint32_t IW = static_cast<uint32_t>(in.dims[3]);
  const uint32_t groups = static_cast<uint32_t>(graph.get_int(groups_id));
  // Transposed weight: [IC, OC/groups, KH, KW].
  const uint32_t KH = static_cast<uint32_t>(weight.dims[2]);
  const uint32_t KW = static_cast<uint32_t>(weight.dims[3]);
  const uint32_t OCpg = static_cast<uint32_t>(weight.dims[1]);
  const uint32_t OC = OCpg * groups;

  if (groups == 0 || IC % groups != 0 || OC % groups != 0) {
    throw std::runtime_error(
        "WebGPU conv_transpose2d: groups must divide IC/OC");
  }
  if (static_cast<uint32_t>(weight.dims[0]) != IC) {
    throw std::runtime_error("WebGPU conv_transpose2d: weight dim0 != IC");
  }
  if (sH == 0 || sW == 0) {
    throw std::runtime_error("WebGPU conv_transpose2d: zero stride");
  }
  if (opH >= sH || opW >= sW) {
    throw std::runtime_error(
        "WebGPU conv_transpose2d: output_padding >= stride");
  }

  // OH = (IH-1)*sH - 2*pH + dH*(KH-1) + output_padding + 1.
  const int64_t oh_v = int64_t(IH - 1) * int64_t(sH) - 2 * int64_t(pH) +
      int64_t(dH) * (int64_t(KH) - 1) + int64_t(opH) + 1;
  const int64_t ow_v = int64_t(IW - 1) * int64_t(sW) - 2 * int64_t(pW) +
      int64_t(dW) * (int64_t(KW) - 1) + int64_t(opW) + 1;
  if (oh_v <= 0 || ow_v <= 0) {
    throw std::runtime_error(
        "WebGPU conv_transpose2d: invalid output geometry");
  }
  const uint32_t OH = static_cast<uint32_t>(oh_v);
  const uint32_t OW = static_cast<uint32_t>(ow_v);

  if (static_cast<uint32_t>(out.dims[0]) != B ||
      static_cast<uint32_t>(out.dims[1]) != OC ||
      static_cast<uint32_t>(out.dims[2]) != OH ||
      static_cast<uint32_t>(out.dims[3]) != OW) {
    throw std::runtime_error("WebGPU conv_transpose2d: output shape mismatch");
  }

  uint64_t out_numel = utils::check_fp32(out, "conv_transpose2d", "output");

  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  // Up-front (throw before any buffer alloc -> no leak-on-throw).
  // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
  // ceiling (FpnNeck @1008^2 out_numel exceeds it). stride_x lets the shader
  // decode i = gid.y*stride_x + gid.x.
  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      utils::checked_u32(out_numel, "conv_transpose2d"),
      kConvTranspose2dWorkgroupSizeX,
      "conv_transpose2d");

  ConvParams params = {};
  params.B = B;
  params.IC = IC;
  params.IH = IH;
  params.IW = IW;
  params.OC = OC;
  params.OH = OH;
  params.OW = OW;
  params.KH = KH;
  params.KW = KW;
  params.sH = sH;
  params.sW = sW;
  params.pH = pH;
  params.pW = pW;
  params.dH = dH;
  params.dW = dW;
  params.groups = groups;
  params.has_bias = has_bias ? 1u : 0u;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ConvParams));
  graph.add_uniform_buffer_bytes(sizeof(ConvParams));

  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);
  auto constants = utils::make_grid_constants(grid);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kConvTranspose2dWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight.buffer,
           weight.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, bias.buffer, bias.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(ConvParams)},
      },
      constants.data(),
      constants.size());

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, grid.count_x, grid.count_y);

  wgpuBufferRelease(uniform_buffer);
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }
}

// aten.convolution.default args: [input, weight, bias, stride, padding,
// dilation, transposed, output_padding, groups, out] (Vulkan Convolution.cpp).
// Direct 2D conv (non-transposed), NCHW fp32, general
// stride/pad/dilation/groups. The fused variant et_vk.conv_with_clamp.default
// is NOT handled (no post-conv activation in the SigLIP patch-embed); it would
// error clearly at load if hit.
void conv2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() != 10) {
    throw std::runtime_error("WebGPU conv2d: expected 10 args (convolution)");
  }
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int stride_id = args.at(3);
  const int padding_id = args.at(4);
  const int dilation_id = args.at(5);
  const int transposed_id = args.at(6);
  const int groups_id = args.at(8);
  const int out_id = args.at(9);

  WGPUDevice device = graph.device();

  if (graph.get_bool(transposed_id)) {
    // C2: fold — transposed conv is a distinct gather kernel, same
    // registration.
    conv_transpose2d_impl(graph, args);
    return;
  }
  // output_padding (args[7]) is only meaningful for transposed conv; require it
  // to be zero so a stale non-transposed export can't silently mis-size output.
  for (int64_t v : graph.get_int_list(args.at(7))) {
    if (v != 0) {
      throw std::runtime_error(
          "WebGPU conv2d: non-zero output_padding unsupported");
    }
  }

  const auto& in = graph.get_tensor(in_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.size() != 4 || weight.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error("WebGPU conv2d: expected 4D input/weight/out");
  }

  // stride/padding/dilation are int lists; PyTorch broadcasts a single value to
  // both spatial dims (e.g. padding="valid" serializes as [0]). Accept size 1
  // (broadcast) or 2 (H, W).
  uint32_t sH, sW, pH, pW, dH, dW;
  utils::parse_hw(graph.get_int_list(stride_id), sH, sW, "conv2d", "stride");
  utils::parse_hw(graph.get_int_list(padding_id), pH, pW, "conv2d", "padding");
  utils::parse_hw(
      graph.get_int_list(dilation_id), dH, dW, "conv2d", "dilation");

  const uint32_t B = static_cast<uint32_t>(in.dims[0]);
  const uint32_t IC = static_cast<uint32_t>(in.dims[1]);
  const uint32_t IH = static_cast<uint32_t>(in.dims[2]);
  const uint32_t IW = static_cast<uint32_t>(in.dims[3]);
  const uint32_t OC = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t KH = static_cast<uint32_t>(weight.dims[2]);
  const uint32_t KW = static_cast<uint32_t>(weight.dims[3]);
  const uint32_t groups = static_cast<uint32_t>(graph.get_int(groups_id));

  if (groups == 0 || OC % groups != 0 || IC % groups != 0) {
    throw std::runtime_error("WebGPU conv2d: groups must divide IC and OC");
  }
  // weight is [OC, IC/groups, KH, KW]; verify the channel dim.
  if (static_cast<uint32_t>(weight.dims[1]) != IC / groups) {
    throw std::runtime_error("WebGPU conv2d: weight in-channels != IC/groups");
  }

  if (sH == 0 || sW == 0) {
    throw std::runtime_error("WebGPU conv2d: zero stride");
  }

  // Compute in signed 64-bit: the numerator goes negative for invalid geometry
  // (e.g. kernel larger than the padded input) and would wrap as unsigned.
  const int64_t oh_num =
      int64_t(IH) + 2 * int64_t(pH) - int64_t(dH) * (int64_t(KH) - 1) - 1;
  const int64_t ow_num =
      int64_t(IW) + 2 * int64_t(pW) - int64_t(dW) * (int64_t(KW) - 1) - 1;
  if (oh_num < 0 || ow_num < 0) {
    throw std::runtime_error(
        "WebGPU conv2d: invalid geometry (kernel > input)");
  }
  const uint32_t OH = static_cast<uint32_t>(oh_num / int64_t(sH)) + 1;
  const uint32_t OW = static_cast<uint32_t>(ow_num / int64_t(sW)) + 1;

  // Validate against the serialized output tensor shape [B, OC, OH, OW].
  if (static_cast<uint32_t>(out.dims[0]) != B ||
      static_cast<uint32_t>(out.dims[1]) != OC ||
      static_cast<uint32_t>(out.dims[2]) != OH ||
      static_cast<uint32_t>(out.dims[3]) != OW) {
    throw std::runtime_error("WebGPU conv2d: output shape mismatch");
  }

  uint64_t out_numel = utils::check_fp32(out, "conv2d", "output");

  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;
  // NCHW's channel dim isn't memory-contiguous, so this is a register-packing
  // vec4 (4 strided scalar loads gathered), not a coalesced load — still cuts
  // the icg loop trip count 4x. Only valid when every group's input-channel
  // count is a multiple of 4 (a real RGB stem's icpg=3 needs the scalar path).
  const uint32_t icpg = IC / groups;
  const bool use_vec4 = (icpg % 4u == 0u);

  // groups==1 non-transposed -> im2col tiled GEMM (reuses the linear tiled-GEMM
  // skeleton: M=OC, N=B*OH*OW, K=IC*KH*KW; input im2col-sampled on the fly,
  // out-of-range -> 0.0 for padding). Canary M4 Pro: 1.1-2.4x over the direct
  // kernel across stem/FPN shapes (biggest on the RGB stem, where the
  // vec4-over-IC path is inert). Grouped/depthwise stay on the direct kernel
  // (grouped GEMM is block-diagonal; mirrors ORT).
  const bool use_gemm = (groups == 1u);

  // Compute the dispatch grid UP-FRONT, before any buffer alloc, so a throw
  // (grid exceeds the device dispatch/tile limit) can't leak the uniform/bias.
  constexpr uint32_t kConv2dGemmTile =
      32u; // fixed @workgroup_size(8,8), TILE=32
  utils::WgCount gemm_grid = {};
  utils::DispatchGrid direct_grid = {};
  if (use_gemm) {
    // 2D tile grid over (N cols, M rows); folds past the 65535 per-dim ceiling.
    gemm_grid = utils::compute_tile_grid_2d(
        device, B * OH * OW, OC, kConv2dGemmTile, "et_vk_conv2d_gemm");
  } else {
    // Adaptive 1D->2D dispatch: wg=clamp(256) + 2D-spill past the 65535 ceiling
    // (the SAM FpnNeck @1008^2 blocker); stride_x decodes
    // i=gid.y*stride_x+gid.x.
    direct_grid = utils::compute_dispatch_grid(
        device,
        utils::checked_u32(out_numel, "conv2d"),
        kConv2dWorkgroupSizeX,
        "et_vk_conv2d");
  }

  ConvParams params = {};
  params.B = B;
  params.IC = IC;
  params.IH = IH;
  params.IW = IW;
  params.OC = OC;
  params.OH = OH;
  params.OW = OW;
  params.KH = KH;
  params.KW = KW;
  params.sH = sH;
  params.sW = sW;
  params.pH = pH;
  params.pW = pW;
  params.dH = dH;
  params.dW = dW;
  params.groups = groups;
  params.has_bias = has_bias ? 1u : 0u;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ConvParams));
  graph.add_uniform_buffer_bytes(sizeof(ConvParams));

  // Dummy 4-byte storage to satisfy the bias binding when None (gated in WGSL).
  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);

  if (use_gemm) {
    // vec4-over-IC is inert for NCHW (strided channel gather), so the GEMM is
    // scalar — ORT skips vec4 for NCHW too.
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kConv2dGemmWGSL,
        {
            {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
            {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
            {2,
             WGPUBufferBindingType_ReadOnlyStorage,
             weight.buffer,
             weight.nbytes},
            {3,
             WGPUBufferBindingType_ReadOnlyStorage,
             bias.buffer,
             bias.nbytes},
            {4,
             WGPUBufferBindingType_Uniform,
             uniform_buffer,
             sizeof(ConvParams)},
        });
    graph.add_dispatch_2d(
        bundle.pipeline, bundle.bind_group, gemm_grid.x, gemm_grid.y);
  } else {
    // Direct conv (grouped/depthwise); vec4-over-IC when icpg%4==0 (register-
    // packed, not coalesced — NCHW's channel stride isn't contiguous).
    auto constants = utils::make_grid_constants(direct_grid);
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        use_vec4 ? kConv2dVec4WGSL : kConv2dWGSL,
        {
            {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
            {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
            {2,
             WGPUBufferBindingType_ReadOnlyStorage,
             weight.buffer,
             weight.nbytes},
            {3,
             WGPUBufferBindingType_ReadOnlyStorage,
             bias.buffer,
             bias.nbytes},
            {4,
             WGPUBufferBindingType_Uniform,
             uniform_buffer,
             sizeof(ConvParams)},
        },
        constants.data(),
        constants.size());
    graph.add_dispatch_2d(
        bundle.pipeline,
        bundle.bind_group,
        direct_grid.count_x,
        direct_grid.count_y);
  }

  wgpuBufferRelease(uniform_buffer);
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.convolution.default, conv2d_impl);
}

} // namespace executorch::backends::webgpu
