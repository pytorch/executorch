/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Preprocess.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Resize output [M, N] based on current fp_input M and packed_weight shape.
// extra_args = { weight_data_tref, fp_input }. Mirrors the style of
// resize_linear_qw_node in QuantizedLinear.cpp.
void resize_q4gsw_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef weight_data = extra_args.at(0);
  const ValueRef fp_input = extra_args.at(1);

  std::vector<int64_t> in_sizes = graph->sizes_of(fp_input);
  std::vector<int64_t> w_sizes = graph->sizes_of(weight_data);

  const int64_t M = utils::val_at(-2, in_sizes);
  // For 4-bit quantization the source weight is [N, K/2].
  const int64_t N = utils::val_at(-2, w_sizes);

  std::vector<int64_t> new_out_sizes;
  if (in_sizes.size() == 2) {
    new_out_sizes = {M, N};
  } else {
    // 3D batched linear: [B, M, K] @ [N, K/2] -> [B, M, N].
    new_out_sizes = {in_sizes.at(0), M, N};
  }
  graph->virtual_resize(output, new_out_sizes);
}

namespace {

//
// Unified dispatch pattern (fp32 + fp16)
//
// Each dtype path emits two execute nodes that cover the full M domain:
//   1. A GEMM DynamicDispatchNode whose global WG self-gates to {0,0,0} at
//      M==1 — handles prefill (M>1) only.
//   2. An adaptive nc-coop GEMV DynamicDispatchNode whose global WG
//      self-gates to {0,0,0} at M!=1 — handles decode (M==1) only.
//
// The framework re-invokes pick_shader_fn / pick_global_wg / pick_local_wg
// on every trigger_resize(), so M transitions across `virtual_resize` are
// routed to the correct node without re-encode beyond what the changed WG
// shape requires.
//
// All participating shaders share a uniform 6-binding layout:
//     (output, fp_input, transposed_input, q4_weights, scales, bias)
// Each shader reads only the bindings it needs; unused bindings compile out
// to zero runtime cost while preserving the shared descriptor set layout.
//
//   - fp32 GEMM       (q4gsw_linear_gemm__w_4x8_nc)         — reads fp_input
//   - fp16 tin GEMM   (q4gsw_linear_gemm__tin__w_4x8_nc)    — reads
//   transposed_input
//   - nc-coop GEMV    (q4gsw_linear_gemv_coop__w_4x8_nc_buffer[_gNwM])
//                                                           — reads fp_input
//
// The fp32 path binds a 0-element TmpTensor into the transposed_input slot
// (never read by any fp32 shader). The fp16 path binds a real
// transposed_input TmpTensor populated by a self-gating transpose preprocess
// dispatch (the preprocess emits no work when M==1).

// Shader picker for the fp32 path — always returns the w_4x8 GEMM kernel.
// M==1 (GEMV) decode is handled exclusively by the adaptive nc-coop GEMV
// sibling node (`add_q4gsw_linear_nc_coop_gemv_node`); this dispatcher's
// global WG self-gates to {0,0,0} when M==1, so the GEMM shader is bound
// but its dispatch is a no-op.
vkapi::ShaderInfo pick_q4gsw_linear_w_4x8_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = "q4gsw_linear_gemm__w_4x8_nc";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Shader picker for the fp16 path — always returns the w_4x8 tin GEMM
// kernel. Same M==1 self-gate semantics as the fp32 picker.
vkapi::ShaderInfo pick_q4gsw_linear_tin_w_4x8_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = "q4gsw_linear_gemm__tin__w_4x8_nc";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

//
// Shape-adaptive nc-coop GEMV picker. Routes M==1 dispatches to one of three
// (NUM_GROUPS, WORKERS_PER_GROUP) decompositions of the cooperative-reduction
// GEMV based on output N. Each variant reads the nc-buffer weight payload
// produced by `prepack_q4_w_4x8_nc_buffer` (shared with the GEMM dispatch — the
// dual nc-Tex2D prepack has been eliminated so the weight is packed only once
// per linear). Only the workgroup geometry and per-lane K-stride differ.
//
// Threshold heuristic chosen from cross-device sweep data (Adreno 750 S24 +
// Adreno 830 S25, all 8 LLM-decode shapes):
//   - N <= 1024: (1, 64) — small N, one WG covers all 8N-tiles efficiently;
//                 wins at K=2048 N=512, K=1024 N=1024, K=3072 N=1024.
//   - N <= 4096: (4, 16) — mid N benefits from finer per-lane K-stride;
//                 wins at K=2048 N=2048, K=8192 N=2048, K=1024 N=2048/3072.
//   - else:      (8, 8) — wide N benefits from multi-tile WGs; wins at
//                 K=2048 N=8192 on both S24/S25 (S25 prefers (16,4) at this
//                 shape, but (8,8) is within 3% there and is robust across
//                 S24 where (16,4) is uniformly worst).
constexpr uint32_t kCoopNgN64 = 1u;
constexpr uint32_t kCoopWpgN64 = 64u;
constexpr uint32_t kCoopNgN4 = 4u;
constexpr uint32_t kCoopWpgN4 = 16u;
constexpr uint32_t kCoopNg8 = 8u;
constexpr uint32_t kCoopWpg8 = 8u;

struct CoopVariant {
  const char* suffix; // append to "q4gsw_linear_gemv_coop__w_4x8_nc_buffer"
  uint32_t num_groups;
  uint32_t workers_per_group;
};

CoopVariant pick_coop_variant_for_N(uint32_t N) {
  if (N <= 1024u) {
    return {"_g1w64", kCoopNgN64, kCoopWpgN64};
  }
  if (N <= 4096u) {
    return {"_g4w16", kCoopNgN4, kCoopWpgN4};
  }
  return {"_g8w8", kCoopNg8, kCoopWpg8};
}

vkapi::ShaderInfo pick_q4gsw_nc_coop_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, graph->sizes_of(out)));

  const CoopVariant v = pick_coop_variant_for_N(N);
  std::string kernel_name = "q4gsw_linear_gemv_coop__w_4x8_nc_buffer";
  kernel_name += v.suffix;
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Global WG for the nc-coop GEMV. Self-gates to {0,0,0} when M != 1 so the
// node is a no-op on prefill (the parallel GEMM dispatch handles M>1).
utils::uvec3 pick_q4gsw_nc_coop_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t> out_sizes = graph->sizes_of(out);
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));
  if (M != 1u) {
    return {0u, 0u, 0u};
  }
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));
  const uint32_t N8 = (N + 7u) / 8u;
  const CoopVariant v = pick_coop_variant_for_N(N);
  const uint32_t wgs_along_x = utils::div_up(N8, v.num_groups);
  return {wgs_along_x, v.num_groups, v.workers_per_group};
}

utils::uvec3 pick_q4gsw_nc_coop_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)global_workgroup_size;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, graph->sizes_of(out)));
  const CoopVariant v = pick_coop_variant_for_N(N);
  return {1u, v.num_groups, v.workers_per_group};
}

} // namespace

// Global WG picker for the fp32 GEMM path. Exposed so the forced-shader test
// selectors (GEMM_W_4X8) can dispatch the same kernel with arbitrary M.
utils::uvec3 pick_q4gsw_linear_gemm_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t> out_sizes = graph->sizes_of(out);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));
  // fp32 GEMM: 4M x 8N per-thread tile.
  return {utils::div_up(N, kGemmTileN), utils::div_up(M, kGemmTileM), 1u};
}

// Local WG picker for the fp32 GEMM path.
utils::uvec3 pick_q4gsw_linear_gemm_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {8u, 8u, 1u};
}

// Global WG picker for the fp16 tin GEMM path.
utils::uvec3 pick_q4gsw_linear_tin_gemm_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t> out_sizes = graph->sizes_of(out);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));
  // fp16 tin GEMM: 8M x 4N per-thread tile. Shader x/y are swapped relative
  // to the fp32 GEMM — x = M tiles, y = N tiles.
  return {utils::div_up(M, kTinGemmTileM), utils::div_up(N, kTinGemmTileN), 1u};
}

// Local WG picker for the fp16 tin GEMM path.
utils::uvec3 pick_q4gsw_linear_tin_gemm_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {1u, 128u, 1u};
}

namespace {

// M==1-gated WG pickers that wrap the shared pickers but self-gate to {0,0,0}
// when M==1. The shape-adaptive nc-coop sibling DynamicDispatchNode handles
// M==1 decode; this gate prevents the GEMM shader from running at M==1 and
// overwriting the nc-coop output. The ungated pickers remain available for
// forced-shader test selectors that need to dispatch GEMM at arbitrary M.
utils::uvec3 pick_q4gsw_linear_gemm_gated_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, graph->sizes_of(out)));
  if (M == 1u) {
    return {0u, 0u, 0u};
  }
  return pick_q4gsw_linear_gemm_global_wg(graph, shader, args, resize_args);
}

utils::uvec3 pick_q4gsw_linear_tin_gemm_gated_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, graph->sizes_of(out)));
  if (M == 1u) {
    return {0u, 0u, 0u};
  }
  return pick_q4gsw_linear_tin_gemm_global_wg(graph, shader, args, resize_args);
}

} // namespace

//
// Prepack helpers
//

// Prepack [N, K/2] uint8 weights into a W_4X8 block-packed nibble buffer
// (each ivec4 covers a 4K x 8N block).
//
// The buffer is allocated with row stride N4_padded (= next-even N4) so that
// the fp32 GEMM shader's 16-byte ivec4 weight load — which spans two
// consecutive (k4, n4) ivec2 tiles along N — never straddles into the next
// k4 row's data. For inputs with N already a multiple of 8 (every existing
// shape), N4 is even and N4_padded == N4, so no extra space is
// allocated and the GEMV reads (which use unpadded N2 = N/2 stride) remain
// bit-identical to the pre-padding layout. For inputs with N % 8 != 0 (e.g.
// N=12, N=20), N4_padded > N4 and the prepack shader fills the OOB n4 tiles
// with the bias-zero pattern (0x88888888u, see the (n < N) branch in
// pack_q4_linear_weight__w_4x8.glsl) — only the fp32 GEMM consumes the
// padded layout and its output store gates n4 + ni < N4, so the OOB tiles
// never affect the output.
ValueRef prepack_q4_w_4x8_nc_buffer(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t N = weight_sizes.at(0);
  const int64_t K = weight_sizes.at(1) * 2;

  VK_CHECK_COND(N % 4 == 0, "N must be a multiple of 4 for W_4X8 uvec2 format");
  VK_CHECK_COND(K % 4 == 0, "K must be a multiple of 4");

  const int64_t K4 = K / 4;
  const int64_t N4 = N / 4;
  // Pad N4 up to the next even value so the fp32 GEMM ivec4 weight load
  // (which spans two consecutive ivec2 tiles along N) never straddles k4
  // rows. No-op for N % 8 == 0.
  const int64_t N4_padded = (N4 + 1) & ~int64_t{1};
  // Each prepack invocation produces one full 4K x 8N block (4 ints in the
  // buffer); N8 = N4_padded / 2 = ceil(N4 / 2).
  const int64_t N8 = N4_padded / 2;

  // Output is a flat int buffer holding 4 * K4 * N8 ints
  // (i.e. K4 * N4_padded ivec2 elements; byte-identical to the legacy 2-tile
  // layout — see pack_q4_linear_weight__w_4x8.glsl).
  const ValueRef packed_weight = graph.add_tensor(
      {K4 * N4_padded * 2}, vkapi::kInt, utils::kBuffer, utils::kWidthPacked);

  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};
  // n4_pitch is unused by the consolidated prepack shader; kept in the push
  // constant block so both buffer and texture2d call sites share an
  // identical layout.
  const int32_t n4_pitch = utils::safe_downcast<int32_t>(N4_padded);

  utils::uvec3 global_wg = {
      utils::safe_downcast<uint32_t>(K4),
      utils::safe_downcast<uint32_t>(N8),
      1u};

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR("pack_q4_linear_weight__w_4x8_nc_buffer"),
      global_wg,
      graph.create_local_wg_size(global_wg),
      weight_data,
      packed_weight,
      {},
      {},
      {graph.sizes_pc_of(packed_weight),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2)),
       PushConstantDataInfo(&n4_pitch, sizeof(int32_t))}));

  return packed_weight;
}

// Prepack [K/gs, N] float scales into a dtype-matched buffer so the GEMM
// shader can read scales as vec4 (fp32) or f16vec4 (fp16) via the binding
// dtype.
ValueRef prepack_q4_scales(
    ComputeGraph& graph,
    const ValueRef weight_scales_data,
    vkapi::ScalarType dtype) {
  ValueRef tensor = graph.add_tensor(
      graph.sizes_of(weight_scales_data),
      dtype,
      utils::kBuffer,
      utils::kWidthPacked);
  add_prepack_standard_node(graph, weight_scales_data, tensor);
  return tensor;
}

//
// Dispatch node builders
//
// Each path emits two execute_nodes:
//   1. GEMM DynamicDispatchNode — self-gates to {0,0,0} when M==1.
//   2. nc-coop GEMV DynamicDispatchNode — self-gates to {0,0,0} when M!=1.
// Together they cover decode (M==1) and prefill (M>1) without re-encode cost,
// since the framework re-runs pick_shader_fn + pick_global_wg on every
// trigger_resize() and re-encodes only when the chosen kernel changes.
//
// The fp16 path additionally requires a transpose preprocess dispatch
// (self-gated to {0,0,0} when M==1) to populate the transposed_input
// TmpTensor that the fp16 tin GEMM reads.
//

// Adds the adaptive nc-coop GEMV sibling dispatch node. The node consumes the
// shared nc-buffer weight prepack (`prepack_q4_w_4x8_nc_buffer`, also used by
// the GEMM dispatch) and the 6-binding layout matching the GEMM dispatch
// (output, fp_input, transposed_input, q4_weights, scales, bias), where
// `transposed_input` is a 0-element dummy (nc-coop never reads it).
//
// Self-gates to {0,0,0} when M != 1 via pick_q4gsw_nc_coop_global_wg, so the
// node is a no-op at prefill. At decode, pick_q4gsw_nc_coop_shader selects
// the nc-buffer coop variant whose (NUM_GROUPS, WORKERS_PER_GROUP) decomp is
// best for the current N. The nc-buffer payload is byte-identical to the
// retired nc-Tex2D payload (see prepack_q4_w_4x8_nc_buffer); only the
// descriptor type and shader weight-fetch path differ, halving the prepacked
// weight memory cost vs the dual-prepack predecessor.
void add_q4gsw_linear_nc_coop_gemv_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef packed_weight,
    const ValueRef weight_data,
    const ValueRef packed_scales,
    const ValueRef packed_bias,
    const uint32_t apply_bias,
    const uint32_t K_val,
    const uint32_t group_size_val,
    const ValueRef output) {
  const vkapi::ScalarType in_dtype = graph.dtype_of(fp_input);

  TmpTensor dummy_transposed_input(
      &graph, {}, in_dtype, utils::kBuffer, utils::kWidthPacked);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_q4gsw_nc_coop_shader,
      pick_q4gsw_nc_coop_global_wg,
      pick_q4gsw_nc_coop_local_wg,
      {{output, vkapi::kWrite},
       {{fp_input,
         dummy_transposed_input.vref,
         packed_weight,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      {apply_bias, K_val, group_size_val},
      {weight_data, fp_input},
      resize_q4gsw_linear_node));
}

void add_q4gsw_linear_w_4x8_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output) {
  // fp32 path. DynamicDispatchNode always binds the fp32 GEMM shader
  // (`q4gsw_linear_gemm__w_4x8_nc`); the gated global WG self-gates the
  // dispatch to {0,0,0} at M==1 so decode is owned by the nc-coop GEMV
  // sibling.
  //
  // A 0-element dummy TmpTensor fills the transposed_input binding slot so
  // that the descriptor set layout matches the tin GEMM shader. The fp32
  // GEMM shader does not reference t_transposed_input.
  const vkapi::ScalarType in_dtype = graph.dtype_of(fp_input);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size_ref);

  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t K = weight_sizes.at(1) * 2;
  const uint32_t K_val = static_cast<uint32_t>(K);

  const ValueRef packed_weight = prepack_q4_w_4x8_nc_buffer(graph, weight_data);
  const ValueRef packed_scales =
      prepack_q4_scales(graph, weight_scales_data, in_dtype);

  // Dummy bias for when bias_data is None — fills the descriptor slot so
  // fewer shader variants are needed.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  // Dummy transposed_input — fills the descriptor slot to match the fp16
  // tin GEMM binding layout. Neither fp32 shader reads this.
  TmpTensor dummy_transposed_input(
      &graph, {}, in_dtype, utils::kBuffer, utils::kWidthPacked);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_q4gsw_linear_w_4x8_shader,
      pick_q4gsw_linear_gemm_gated_global_wg,
      pick_q4gsw_linear_gemm_local_wg,
      {{output, vkapi::kWrite},
       {{fp_input,
         dummy_transposed_input.vref,
         packed_weight,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      {apply_bias, K_val, static_cast<uint32_t>(group_size_val)},
      {weight_data, fp_input},
      resize_q4gsw_linear_node));

  // Sibling adaptive nc-coop GEMV — handles M==1; no-ops at prefill.
  // Shares the nc-buffer weight prepack with the GEMM dispatch above so the
  // weight is packed only once per linear (vs the prior dual nc-buffer +
  // nc-Tex2D prepack).
  add_q4gsw_linear_nc_coop_gemv_node(
      graph,
      fp_input,
      packed_weight,
      weight_data,
      packed_scales,
      packed_bias,
      apply_bias,
      K_val,
      static_cast<uint32_t>(group_size_val),
      output);
}

void add_q4gsw_linear_tin_w_4x8_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output) {
  // fp16 path. Two execute nodes:
  //   1. Transpose preprocess — self-gates to {0,0,0} when M==1, populates
  //      the transposed_input TmpTensor for the tin GEMM shader.
  //   2. DynamicDispatchNode binding the fp16 tin GEMM shader
  //      (`q4gsw_linear_gemm__tin__w_4x8_nc`); the gated global WG self-gates
  //      the dispatch to {0,0,0} at M==1 so decode is owned by the nc-coop
  //      GEMV sibling.
  const vkapi::ScalarType in_dtype = graph.dtype_of(fp_input);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size_ref);

  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t K = weight_sizes.at(1) * 2;
  const uint32_t K_val = static_cast<uint32_t>(K);

  const ValueRef packed_weight = prepack_q4_w_4x8_nc_buffer(graph, weight_data);
  const ValueRef packed_scales =
      prepack_q4_scales(graph, weight_scales_data, in_dtype);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  std::vector<int64_t> out_sizes = graph.sizes_of(output);
  const uint32_t M_val =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));

  // Allocate the transposed-input temp tensor using the current M. The
  // transpose dispatch self-gates on M==1 so the tensor is simply unused in
  // the GEMV case (its contents are not read by the GEMV shader). A later
  // virtual_resize that grows M past this allocation will be rejected by
  // vTensor::check_sizes before the transpose shader can run, so the graph
  // must be built with the largest expected M.
  const int64_t M4 = (static_cast<int64_t>(M_val) + 3) / 4;
  TmpTensor transposed_input(
      &graph,
      {static_cast<int64_t>(K_val) * M4 * 4},
      in_dtype,
      utils::kBuffer,
      utils::kWidthPacked);
  // Preprocess transpose — self-gates when M==1 (see Preprocess.cpp). Emits
  // no work for the GEMV case so the tensor is simply unread.
  add_transpose_cast_contig_to_vectorized_node(
      graph, fp_input, transposed_input.vref);

  // Precompute kernel names.
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_q4gsw_linear_tin_w_4x8_shader,
      pick_q4gsw_linear_tin_gemm_gated_global_wg,
      pick_q4gsw_linear_tin_gemm_local_wg,
      {{output, vkapi::kWrite},
       {{fp_input,
         transposed_input.vref,
         packed_weight,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      {apply_bias, K_val, static_cast<uint32_t>(group_size_val)},
      {weight_data, fp_input},
      resize_q4gsw_linear_node));

  // Sibling adaptive nc-coop GEMV — handles M==1; no-ops at prefill.
  // Shares the nc-buffer weight prepack with the TIN GEMM dispatch above so
  // the weight is packed only once per linear (vs the prior dual nc-buffer +
  // nc-Tex2D prepack).
  add_q4gsw_linear_nc_coop_gemv_node(
      graph,
      fp_input,
      packed_weight,
      weight_data,
      packed_scales,
      packed_bias,
      apply_bias,
      K_val,
      static_cast<uint32_t>(group_size_val),
      output);
}

void q4gsw_linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size_ref = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx);

  // Dtype-branched dispatch. Within each dtype, a single DynamicDispatchNode
  // switches between GEMM and GEMV via pick_shader_fn based on the current M.
  const vkapi::ScalarType in_dtype = graph.dtype_of(fp_input);

  if (in_dtype == vkapi::kFloat) {
    add_q4gsw_linear_w_4x8_node(
        graph,
        fp_input,
        weight_data,
        weight_scales_data,
        group_size_ref,
        bias_data,
        output);
  } else {
    add_q4gsw_linear_tin_w_4x8_node(
        graph,
        fp_input,
        weight_data,
        weight_scales_data,
        group_size_ref,
        bias_data,
        output);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q4gsw_linear.default, q4gsw_linear);
  VK_REGISTER_OP(et_vk.linear_q4gsw.default, q4gsw_linear);
}

} // namespace vkcompute
