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
#include <executorch/backends/webgpu/runtime/ops/conv1d_dw/conv1d_dw.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/Conv2d.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_shmem_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_vec4_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d_gemm_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

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
  uint32_t has_clamp; // conv2d_gemm / conv2d_shmem only
  float out_min;
  float out_max;
};
static_assert(
    sizeof(ConvParams) == 80,
    "ConvParams must be 80 bytes (16-mult)");

// conv2d_gemm.wgsl: fixed @workgroup_size(8,8), TILE=32.
constexpr uint32_t kConv2dGemmTile = 32u;

// Below these a 32x32x32 GEMM tile is mostly padding, or the GEMM launches too
// few tiles to fill the GPU, and the direct or gather kernel is as fast or
// faster.
constexpr uint64_t kGemmMinM = 8u;
constexpr uint64_t kGemmMinK = 16u;
constexpr uint64_t kGemmMinWorkgroups = 256u;

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
  // Guard a u32 underflow in (IH-1)/(IW-1) below, and a zero dilation/kernel
  // that would mis-size the output geometry.
  if (IH == 0 || IW == 0) {
    throw std::runtime_error("WebGPU conv_transpose2d: zero input spatial dim");
  }
  if (dH == 0 || dW == 0 || KH == 0 || KW == 0) {
    throw std::runtime_error(
        "WebGPU conv_transpose2d: zero dilation or kernel dim");
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

  // Non-overlapping taps (stride == kernel, e.g. 2x upsampling) reduce to a
  // GEMM (M = OC*KH*KW, N = input pixels, K = IC) plus a depth-to-space
  // scatter. Other shapes, GEMMs too small to pay off, and GEMM tile grids past
  // the dispatch limit use the per-output gather (which folds its grid).
  constexpr uint32_t kConvTranspose2dGemmTile = 32u; // TILE in the shader
  const uint64_t max_wgs = utils::queried_max_workgroups(device);
  const uint64_t gemm_m = uint64_t(OC) * KH * KW;
  const uint64_t gemm_cols =
      utils::div_up<uint64_t>(uint64_t(B) * IH * IW, kConvTranspose2dGemmTile);
  const uint64_t gemm_rows =
      utils::div_up<uint64_t>(gemm_m, kConvTranspose2dGemmTile);
  const bool use_gemm = groups == 1 && sH == KH && sW == KW && pH == 0 &&
      pW == 0 && dH == 1 && dW == 1 && opH == 0 && opW == 0 &&
      gemm_m >= kGemmMinM && IC >= kGemmMinK &&
      gemm_cols * gemm_rows >= kGemmMinWorkgroups && gemm_cols <= max_wgs &&
      gemm_rows <= max_wgs;

  // Up-front (throw before any buffer alloc -> no leak-on-throw).
  utils::WgCount gemm_grid = {};
  utils::DispatchGrid grid = {};
  if (use_gemm) {
    gemm_grid = utils::compute_tile_grid_2d(
        device,
        utils::checked_u32(
            uint64_t(B) * uint64_t(IH) * uint64_t(IW), "conv_transpose2d_gemm"),
        OC * KH * KW,
        kConvTranspose2dGemmTile,
        "conv_transpose2d_gemm");
  } else {
    // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
    // ceiling for large outputs. stride_x lets the shader decode
    // i = gid.y*stride_x + gid.x.
    grid = utils::compute_dispatch_grid(
        device,
        utils::checked_u32(out_numel, "conv_transpose2d"),
        kConvTranspose2dWorkgroupSizeX,
        "conv_transpose2d");
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

  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);
  const std::vector<utils::BindingSpec> bindings = {
      {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
      {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
      {2, WGPUBufferBindingType_ReadOnlyStorage, weight.buffer, weight.nbytes},
      {3, WGPUBufferBindingType_ReadOnlyStorage, bias.buffer, bias.nbytes},
      {4, WGPUBufferBindingType_Uniform, uniform_buffer, sizeof(ConvParams)},
  };

  if (use_gemm) {
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device, kConvTranspose2dGemmWGSL, bindings);
    graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         gemm_grid.x,
         "conv_transpose2d_gemm",
         gemm_grid.y});
  } else {
    auto constants = utils::make_grid_constants(grid);
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kConvTranspose2dWGSL,
        bindings,
        constants.data(),
        constants.size());
    graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         grid.count_x,
         "conv_transpose2d",
         grid.count_y});
  }

  wgpuBufferRelease(uniform_buffer);
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }
}

} // namespace

// aten.convolution.default args: [input, weight, bias, stride, padding,
// dilation, transposed, output_padding, groups, out] (Vulkan Convolution.cpp).
// Direct 2D conv (non-transposed), NCHW fp32, general
// stride/pad/dilation/groups.
bool add_conv2d(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    std::optional<ConvClamp> clamp,
    bool can_decline) {
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

  // aten.convolution.default covers conv1d (3D) and conv2d (4D); the registry
  // is first-wins, so one handler must serve both ranks. Route 3D to conv1d.
  if (graph.get_tensor(in_id).dims.size() == 3) {
    conv1d_dispatch(graph, args);
    return true;
  }

  WGPUDevice device = graph.device();

  if (graph.get_bool(transposed_id)) {
    // C2: fold — transposed conv is a distinct gather kernel, same
    // registration.
    conv_transpose2d_impl(graph, args);
    return true;
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
  if (dH == 0 || dW == 0 || KH == 0 || KW == 0) {
    throw std::runtime_error("WebGPU conv2d: zero dilation or kernel dim");
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
  if (clamp && !use_gemm) {
    throw std::runtime_error("WebGPU conv2d: fused clamp needs groups == 1");
  }
  // Small GEMMs (M = OC, N = output pixels, K = IC*KH*KW) leave to a caller's
  // direct kernel, which is faster there, before conv2d_shmem is considered.
  const uint64_t max_wgs = utils::queried_max_workgroups(device);
  const uint64_t gemm_cols =
      utils::div_up<uint64_t>(uint64_t(B) * OH * OW, kConv2dGemmTile);
  const uint64_t gemm_rows = utils::div_up<uint64_t>(OC, kConv2dGemmTile);
  const bool gemm_pays_off = OC >= kGemmMinM &&
      uint64_t(IC) * KH * KW >= kGemmMinK &&
      gemm_cols * gemm_rows >= kGemmMinWorkgroups;
  if (can_decline && (!use_gemm || !gemm_pays_off)) {
    return false;
  }

  // Compute the dispatch grid UP-FRONT, before any buffer alloc, so a throw
  // (grid exceeds the device dispatch/tile limit) can't leak the uniform/bias.
  // groups==1 with more than one tap -> conv2d_shmem, which stages each tile's
  // input in shared memory once and reuses it across all KH*KW taps (the
  // im2col GEMM refetches it per tap). Narrower channel tiles for small OC.
  const uint32_t wg_m = OC > 16u ? 8u : OC > 8u ? 4u : OC > 4u ? 2u : 1u;
  const uint32_t tile_m = 4u * wg_m;
  constexpr uint32_t kShmemPpt = 8u;
  constexpr uint32_t kShmemTw = 16u;
  const uint32_t th = (64u / wg_m) * kShmemPpt / kShmemTw;
  // 64-bit, and each footprint side bounded by the storage's float capacity
  // before any product, so large dilation or padding cannot wrap into a small
  // size.
  constexpr uint64_t kShmemBytes = 16384u;
  const uint64_t footprint_h =
      uint64_t(th - 1u) * sH + uint64_t(KH - 1u) * dH + 1u;
  const uint64_t footprint_w =
      uint64_t(kShmemTw - 1u) * sW + uint64_t(KW - 1u) * dW + 1u;
  const bool footprint_fits = footprint_h <= kShmemBytes / sizeof(float) &&
      footprint_w <= kShmemBytes / sizeof(float);
  auto shmem_bytes = [&](uint64_t ict) {
    return ict * (footprint_h * footprint_w + uint64_t(KH) * KW * tile_m) *
        sizeof(float);
  };
  // Deepest input-channel chunk, no deeper than IC, that fits WebGPU's 16 KiB
  // default.
  uint32_t ict = std::min(8u, IC);
  while (footprint_fits && ict > 1u && shmem_bytes(ict) > kShmemBytes) {
    ict /= 2u;
  }
  // A shmem tile is large (up to 32 channels x 64 pixels), so small outputs
  // launch too few workgroups to fill the GPU; there the GEMM's smaller tiles
  // are as fast or faster. Its grid is not folded, so it must also fit one
  // dispatch; otherwise the GEMM takes the conv.
  constexpr uint64_t kShmemMinWorkgroups = 128u;
  const uint64_t shmem_grid_x = utils::div_up<uint64_t>(OW, kShmemTw) *
      utils::div_up<uint64_t>(OC, tile_m);
  const uint64_t shmem_grid_y = utils::div_up<uint64_t>(OH, th) * B;
  const bool use_shmem = use_gemm && KH * KW > 1u && footprint_fits &&
      shmem_bytes(ict) <= kShmemBytes &&
      shmem_grid_x * shmem_grid_y >= kShmemMinWorkgroups &&
      shmem_grid_x <= max_wgs && shmem_grid_y <= max_wgs;

  // Declining also keeps a caller's folded dispatch for outputs whose GEMM
  // grid would not fit.
  if (can_decline && !use_shmem &&
      (gemm_cols > max_wgs || gemm_rows > max_wgs)) {
    return false;
  }

  utils::WgCount gemm_grid = {};
  utils::DispatchGrid direct_grid = {};
  if (use_shmem) {
    gemm_grid = {
        static_cast<uint32_t>(shmem_grid_x),
        static_cast<uint32_t>(shmem_grid_y)};
  } else if (use_gemm) {
    // 2D tile grid over (N cols, M rows); folds past the 65535 per-dim ceiling.
    gemm_grid = utils::compute_tile_grid_2d(
        device,
        utils::checked_u32(
            uint64_t(B) * uint64_t(OH) * uint64_t(OW), "et_vk_conv2d_gemm"),
        OC,
        kConv2dGemmTile,
        "et_vk_conv2d_gemm");
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
  if (clamp) {
    params.has_clamp = 1u;
    params.out_min = clamp->min;
    params.out_max = clamp->max;
  }

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ConvParams));
  graph.add_uniform_buffer_bytes(sizeof(ConvParams));

  // Dummy 4-byte storage to satisfy the bias binding when None (gated in WGSL).
  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);

  if (use_shmem) {
    const std::array<std::pair<const char*, uint32_t>, 8> values = {{
        {"WG_M", wg_m},
        {"PPT", kShmemPpt},
        {"TW", kShmemTw},
        {"ICT", ict},
        {"KH", KH},
        {"KW", KW},
        {"HH", static_cast<uint32_t>(footprint_h)},
        {"HW", static_cast<uint32_t>(footprint_w)},
    }};
    std::array<WGPUConstantEntry, 8> constants = {};
    for (size_t i = 0; i < values.size(); i++) {
      constants[i].key = {values[i].first, WGPU_STRLEN};
      constants[i].value = values[i].second;
    }
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kConv2dShmemWGSL,
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
    graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         gemm_grid.x,
         "conv2d_shmem",
         gemm_grid.y});
  } else if (use_gemm) {
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
    graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         gemm_grid.x,
         "conv2d_gemm",
         gemm_grid.y});
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
  return true;
}

namespace {

void conv2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_conv2d(graph, args, std::nullopt);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.convolution.default, conv2d_impl);
}

} // namespace executorch::backends::webgpu
