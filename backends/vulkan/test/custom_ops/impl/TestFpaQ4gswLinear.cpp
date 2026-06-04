/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Preprocess.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

// File-scoped enum mirroring the previously-proposed Q4gswLinearKernelKind.
// Kept internal to the test op so that production code stays untouched.
enum class TestKernelKind {
  PROD, // Dtype-based picker: fp32 -> w_4x8, fp16 -> tin_w_4x8.
  GEMM_W_4X8, // Force non-tin GEMM (reads fp_input directly).
  GEMM_TIN_W_4X8, // Force tin GEMM (transposed input preprocess emitted).
  GEMV_W_4X8, // Force gemv with subgroup broadcast.
  GEMV_W_4X8_NOSG, // Force gemv without subgroup broadcast.
  LEGACY, // Legacy in-prod q4gsw linear (et_vk.linear_q4gsw.default).
  GEMV_COOP_W_4X8_NC_BUFFER, // coop GEMV with nc Buffer weight (prod nc-buf
                             // prepack). Equivalent to the _g1w64 variant
                             // (NUM_GROUPS=1, WORKERS_PER_GROUP=64).
  // Forced coop nc-buffer GEMV reduction-decomposition variants. The
  // production picker (pick_coop_variant_for_N in Q4gswLinear.cpp) selects
  // among these based on output N: N<=1024 -> g1w64, N<=4096 -> g4w16, else
  // g8w8. The PERF-sized N where g4w16 / g8w8 are normally chosen exceeds
  // kRefDimSizeLimit, so the reference impl is skipped there; these forced
  // selectors let the same (NUM_GROUPS, WORKERS_PER_GROUP) decompositions be
  // validated at small N where the reference runs.
  GEMV_COOP_W_4X8_NC_BUFFER_G1W64, // NUM_GROUPS=1, WORKERS_PER_GROUP=64.
  GEMV_COOP_W_4X8_NC_BUFFER_G4W16, // NUM_GROUPS=4, WORKERS_PER_GROUP=16.
  GEMV_COOP_W_4X8_NC_BUFFER_G8W8, // NUM_GROUPS=8, WORKERS_PER_GROUP=8.
};

// Map the selector int + table (gemm vs gemv) to a TestKernelKind.
//
// is_gemv = false (gemm op):
//   0  -> PROD, 1 -> GEMM_W_4X8, 2 -> GEMM_TIN_W_4X8, 3 -> LEGACY.
//
// is_gemv = true (gemv op):
//   0  -> PROD, 1 -> GEMV_W_4X8, 2 -> GEMV_W_4X8_NOSG, 3 -> LEGACY,
//   13 -> GEMV_COOP_W_4X8_NC_BUFFER (coop GEMV reusing the production
//          nc-buffer prepack — same weight format used by W_4X8 GEMM/TIN GEMM
//          / sg-GEMV; tests whether a single prepack can serve both prefill
//          and decode). Equivalent to the _g1w64 reduction decomposition.
//   14 -> GEMV_COOP_W_4X8_NC_BUFFER_G1W64 (force NUM_GROUPS=1, WPG=64).
//   15 -> GEMV_COOP_W_4X8_NC_BUFFER_G4W16 (force NUM_GROUPS=4, WPG=16).
//   16 -> GEMV_COOP_W_4X8_NC_BUFFER_G8W8  (force NUM_GROUPS=8, WPG=8).
//
// Selectors 14-16 pin the coop nc-buffer GEMV to an explicit reduction
// decomposition regardless of N, so the g4w16 / g8w8 variants (otherwise only
// chosen by the production picker at PERF-sized N where the reference impl is
// skipped) can be ACCU-validated at small N. The production picker behavior
// (pick_coop_variant_for_N) is unaffected — these are test-only forced paths.
//
// Selector 3 (LEGACY) dispatches the in-prod legacy linear path registered as
// et_vk.linear_q4gsw.default in QuantizedLinear.cpp. It uses a different
// prepack (pack_q4_linear_weight) and different shaders (linear_q4gsw_tiled_*
// / linear_q4gsw_coop_*); it picks GEMM vs GEMV internally based on input M.
TestKernelKind selector_to_kind(int32_t selector, bool is_gemv) {
  if (is_gemv) {
    switch (selector) {
      case 0:
        return TestKernelKind::PROD;
      case 1:
        return TestKernelKind::GEMV_W_4X8;
      case 2:
        return TestKernelKind::GEMV_W_4X8_NOSG;
      case 3:
        return TestKernelKind::LEGACY;
      case 13:
        return TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER;
      case 14:
        return TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G1W64;
      case 15:
        return TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16;
      case 16:
        return TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8;
      default:
        return TestKernelKind::PROD;
    }
  }
  switch (selector) {
    case 0:
      return TestKernelKind::PROD;
    case 1:
      return TestKernelKind::GEMM_W_4X8;
    case 2:
      return TestKernelKind::GEMM_TIN_W_4X8;
    case 3:
      return TestKernelKind::LEGACY;
    default:
      return TestKernelKind::PROD;
  }
}

// Returns the fixed base kernel name for a given forced kind.
const char* forced_kind_base_name(TestKernelKind kind) {
  switch (kind) {
    case TestKernelKind::GEMM_W_4X8:
      return "q4gsw_linear_gemm__w_4x8_nc";
    case TestKernelKind::GEMM_TIN_W_4X8:
      return "q4gsw_linear_gemm__tin__w_4x8_nc";
    case TestKernelKind::GEMV_W_4X8:
      return "q4gsw_linear_gemv__w_4x8_nc";
    case TestKernelKind::GEMV_W_4X8_NOSG:
      return "q4gsw_linear_gemv__w_4x8_nc_nosg";
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER:
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G1W64:
      return "q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g1w64";
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16:
      return "q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g4w16";
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8:
      return "q4gsw_linear_gemv_coop__w_4x8_nc_buffer_g8w8";
    case TestKernelKind::PROD:
    case TestKernelKind::LEGACY:
    default:
      return "";
  }
}

// Build a picker that ignores M and always returns the forced shader.
// Storage + dtype suffixes are appended at dispatch time.
template <TestKernelKind KIND>
vkapi::ShaderInfo pick_forced_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = forced_kind_base_name(KIND);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Picker for the new coop kc variant. The weight is bound as a Tex2D image
// (kc dense form) but the kernel naming convention only tags IO storage; we
// therefore append only the IO (output) storage suffix + dtype.
template <TestKernelKind KIND>
vkapi::ShaderInfo pick_forced_shader_coop_kc(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = forced_kind_base_name(KIND);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Coop GEMV NUM_GROUPS / WORKERS_PER_GROUP knobs. The chosen pair must agree
// with the bound shader variant's GLSL codegen params — the shader's shared
// memory is sized NUM_GROUPS * WORKERS_PER_GROUP and the K-loop strides by
// WORKERS_PER_GROUP, so a dispatch geometry mismatch produces wrong results.
// Templating the WG pickers on the pair lets each forced variant selector
// (g1w64 / g4w16 / g8w8) dispatch matching geometry. (NUM_GROUPS=1,
// WORKERS_PER_GROUP=64) reproduces the original dispatch (LWG=(1,1,64), one WG
// per n8 tile = 8 outputs).
//
// Global WG picker for the coop GEMV. Each WG hosts NUM_GROUPS independent
// worker groups (each producing 8 outputs); WGs along x = ceil(N8 /
// NUM_GROUPS). The framework computes num_WGs = div_up(global, local), so the
// global x-axis is set to that count directly (with local.x == 1).
template <uint32_t NUM_GROUPS, uint32_t WORKERS_PER_GROUP>
utils::uvec3 pick_q4gsw_coop_global_wg(
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
  const uint32_t N8 = (N + 7u) / 8u;
  const uint32_t wgs_along_x = utils::div_up(N8, NUM_GROUPS);
  return {wgs_along_x, NUM_GROUPS, WORKERS_PER_GROUP};
}

// Local WG picker for the coop GEMV — LWG=(1, NUM_GROUPS, WORKERS_PER_GROUP).
template <uint32_t NUM_GROUPS, uint32_t WORKERS_PER_GROUP>
utils::uvec3 pick_q4gsw_coop_local_wg(
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
  return {1u, NUM_GROUPS, WORKERS_PER_GROUP};
}

// Spec-constant LWG values for the q4gsw_linear_gemv__w_4x8_nc[_nosg] shaders
// (now shipped from test/custom_ops/glsl/). The sg variant pins subgroupSize
// to 64 via VK_EXT_subgroup_size_control; the nosg variant uses shared-mem
// reduction so the lane count is purely an LWG choice. Both use 4 subgroups
// per workgroup.
constexpr uint32_t kGemvSubgroupSize = 64u;
constexpr uint32_t kGemvNumSubgroups = 4u;

// WG pickers for the legacy sg/nosg GEMV shaders. Used only by test selectors
// 1 (GEMV_W_4X8) and 2 (GEMV_W_4X8_NOSG); the production dispatcher never
// references these shaders.
utils::uvec3 pick_q4gsw_legacy_gemv_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, graph->sizes_of(out)));
  // Each thread owns one row-pair along x; y-dim splits K-blocks across waves.
  return {N / 2u, kGemvNumSubgroups, 1u};
}

utils::uvec3 pick_q4gsw_legacy_gemv_local_wg(
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
  return {kGemvSubgroupSize, kGemvNumSubgroups, 1u};
}

//
// Legacy q4gsw linear dispatch — copy of the implementation deleted from
// runtime/graph/ops/impl/QuantizedLinear.cpp by the W_4X8 commit
// (6d1fa80b3c79). Resurrected here so selector 3 (LEGACY) exercises the legacy
// `linear_q4gsw_tiled` / `linear_q4gsw_coop` shaders + `pack_q4_linear_weight`
// prepack directly, without depending on a registered production op. Trimmed
// to the q4gsw weight-only branch (no 8-bit / no activation-quant path).
//

void legacy_q4gsw_resize_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  ValueRef output = args.at(0).refs.at(0);
  ValueRef fp_input = args.at(1).refs.at(0);
  ValueRef weight_data = extra_args.at(1);

  std::vector<int64_t> mat1_sizes = graph->sizes_of(fp_input);
  std::vector<int64_t> mat2_sizes = graph->sizes_of(weight_data);

  const int64_t out_cols = utils::val_at(-2, mat1_sizes);
  const int64_t out_rows = utils::val_at(-2, mat2_sizes);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1_sizes.size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  graph->virtual_resize(output, new_out_sizes);
}

utils::uvec3 legacy_q4gsw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  // width
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));
  // height
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));

  // For 4-bit weights, each output tile contains 8 columns
  uint32_t N_per_tile = 8;
  uint32_t M_per_tile = 4;
  if (shader.kernel_name.find("coop") != std::string::npos) {
    M_per_tile = 1;
  }

  const uint32_t num_N_tiles = utils::div_up(N, N_per_tile);
  const uint32_t num_M_tiles = utils::div_up(M, M_per_tile);

  return {num_N_tiles, num_M_tiles, 1};
}

utils::uvec3 legacy_q4gsw_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 1, 64};
  }
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

vkapi::ShaderInfo legacy_q4gsw_pick_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_int_weight = args.at(1).refs.at(1);

  const bool is_gemv_case = is_gemv(graph, fp_input);

  std::string kernel_name = "linear_q4gsw";
  kernel_name += is_gemv_case ? "_coop" : "_tiled";

  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(
      kernel_name, graph->storage_type_of(packed_int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));

  return VK_KERNEL_FROM_STR(kernel_name);
}

// Legacy 4-bit weight prepack — populates the [num_blocks_N, num_blocks_K * 4]
// tensor used by linear_q4gsw_tiled / linear_q4gsw_coop. Uses
// pack_q4_linear_weight, NOT the W_4X8 nc-pair prepack.
ValueRef legacy_prepack_q4gsw_weight(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  const int64_t qmat2_width = qmat2_orig_sizes.at(ndim - 1);
  const int64_t qmat2_height = qmat2_orig_sizes.at(ndim - 2);

  // For 4-bit quantization, source weight has shape [N, K/2]; each byte
  // contains 2 nibbles.
  const int64_t K = qmat2_width * 2;
  const int64_t N = qmat2_height;

  VK_CHECK_COND(K % 8 == 0);

  // 4-bit blocks: 8 rows of N per block, 4 columns of K per block.
  const int64_t N_per_block = 8;
  const int64_t K_per_block = 4;

  const int64_t num_blocks_K = utils::div_up(K, K_per_block);
  const int64_t num_blocks_N = utils::div_up(N, N_per_block);

  // Layout for the coop GEMV path: packed_weights[n8][k4] (no transposition).
  const int64_t output_height = num_blocks_N;
  const int64_t output_width = num_blocks_K * 4;

  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  std::vector<int64_t> qmat2_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }

  std::string kernel_name = "pack_q4_linear_weight";
  add_storage_type_suffix(kernel_name, storage_type);

  // Reuse the prepack cache so repeated test invocations don't re-run the
  // prepack shader against the same weight TensorRef.
  ValueRef cached = graph.get_cached_prepack(qmat2_data, kernel_name);
  if (is_valid(cached)) {
    return cached;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkapi::kInt, storage_type, utils::kWidthPacked);

  // 4-bit prepack: each thread writes two adjacent blocks along K.
  utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(utils::div_up(num_blocks_K, int64_t(2))),
      utils::safe_downcast<uint32_t>(num_blocks_N),
      1u};

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      qmat2_data,
      qmat2,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(qmat2),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  graph.cache_prepack(qmat2_data, kernel_name, qmat2);
  return qmat2;
}

// Replacement for the deleted et_vk.linear_q4gsw.default registration. Mirrors
// the legacy q4gsw weight-only path: pack_q4_linear_weight prepack + buffer
// scales/bias prepack + a single DynamicDispatchNode whose pick_shader_fn
// picks coop (M==1) or tiled (M>1) at trigger_resize().
void add_legacy_q4gsw_linear_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output) {
  std::vector<int64_t> input_sizes = graph.sizes_of(fp_input);
  const int64_t K = utils::val_at(-1, input_sizes);
  // K must be a multiple of 4 so vec4 input loads are aligned.
  VK_CHECK_COND(K % 4 == 0);

  const ValueRef packed_weight =
      legacy_prepack_q4gsw_weight(graph, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  const int32_t group_size_val = graph.extract_scalar<int32_t>(group_size_ref);
  const int32_t K4_per_group = utils::div_up(group_size_val, int32_t(4));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      legacy_q4gsw_pick_shader,
      legacy_q4gsw_global_wg_size,
      legacy_q4gsw_local_wg_size,
      // Inputs and Outputs (legacy 5-binding layout)
      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {apply_bias, K4_per_group},
      // Resize args. extra_args.at(0) is unused (was the "is_4bit_flag"
      // gate in the legacy multi-precision dispatcher); keep
      // weight_data at index 1 so resize logic can read sizes_of(weight_data).
      {kDummyValueRef, weight_data},
      legacy_q4gsw_resize_linear_node));
}

// Forced-shader dispatch for the coop GEMV nc-Buffer variants (selectors
// 13-16). Reuses the production nc-buffer prepack (shared with W_4X8 GEMM /
// TIN GEMM / sg-GEMV via the prepack cache) — same SSBO payload, tests
// single-prepack viability across prefill + decode.
//
// `kind` selects which (NUM_GROUPS, WORKERS_PER_GROUP) reduction decomposition
// to pin: g1w64 -> LWG=(1,1,64) (one WG per n8 tile), g4w16 -> LWG=(1,4,16),
// g8w8 -> LWG=(1,8,8). The bound shader variant and the dispatch geometry are
// kept in sync (both keyed on `kind`) so the shared-memory layout the shader
// bakes in matches the launched workgroup shape. This forces a fixed
// decomposition regardless of N, mirroring what the production picker would
// pick at a given N but at a shape small enough for the reference impl to run.
void add_q4gsw_linear_coop_kc_forced_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output,
    TestKernelKind kind) {
  const vkapi::ScalarType in_dtype = graph.dtype_of(fp_input);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size_ref);

  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t K = weight_sizes.at(1) * 2;
  const uint32_t K_val = static_cast<uint32_t>(K);

  const ValueRef packed_weight_kc =
      prepack_q4_w_4x8_nc_buffer(graph, weight_data);
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

  TmpTensor dummy_transposed_input(
      &graph, {}, in_dtype, utils::kBuffer, utils::kWidthPacked);

  using PickShaderFn = vkapi::ShaderInfo (*)(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickWgFn = utils::uvec3 (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickLocalWgFn = utils::uvec3 (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const utils::uvec3&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);

  PickShaderFn pick_shader = nullptr;
  PickWgFn pick_global = nullptr;
  PickLocalWgFn pick_local = nullptr;

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (kind) {
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER:
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G1W64:
      pick_shader =
          pick_forced_shader_coop_kc<TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER>;
      pick_global = pick_q4gsw_coop_global_wg<1u, 64u>;
      pick_local = pick_q4gsw_coop_local_wg<1u, 64u>;
      break;
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16:
      pick_shader = pick_forced_shader_coop_kc<
          TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16>;
      pick_global = pick_q4gsw_coop_global_wg<4u, 16u>;
      pick_local = pick_q4gsw_coop_local_wg<4u, 16u>;
      break;
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8:
      pick_shader = pick_forced_shader_coop_kc<
          TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8>;
      pick_global = pick_q4gsw_coop_global_wg<8u, 8u>;
      pick_local = pick_q4gsw_coop_local_wg<8u, 8u>;
      break;
    default:
      VK_THROW("add_q4gsw_linear_coop_kc_forced_node: non-coop kind");
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader,
      pick_global,
      pick_local,
      {{output, vkapi::kWrite},
       {{fp_input,
         dummy_transposed_input.vref,
         packed_weight_kc,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      {apply_bias, K_val, static_cast<uint32_t>(group_size_val)},
      {weight_data, fp_input},
      resize_q4gsw_linear_node));
}

// Forced-shader dispatch path. Used only by selectors 1 and 2.
void add_q4gsw_linear_forced_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output,
    TestKernelKind kind) {
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

  // GEMM_TIN_W_4X8 needs a real transposed_input + transpose preprocess
  // dispatch. Other forced kinds use a 0-element dummy — the bound shader
  // never reads the slot.
  const bool need_transpose = (kind == TestKernelKind::GEMM_TIN_W_4X8);

  std::vector<int64_t> in_sizes = graph.sizes_of(fp_input);
  const uint32_t M_val =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, in_sizes));
  const int64_t M4 = (static_cast<int64_t>(M_val) + 3) / 4;

  TmpTensor dummy_transposed_input(
      &graph, {}, in_dtype, utils::kBuffer, utils::kWidthPacked);
  TmpTensor real_transposed_input(
      &graph,
      {static_cast<int64_t>(K_val) * M4 * 4},
      in_dtype,
      utils::kBuffer,
      utils::kWidthPacked);

  ValueRef transposed_input_ref;
  if (need_transpose) {
    transposed_input_ref = real_transposed_input.vref;
    add_transpose_cast_contig_to_vectorized_node(
        graph, fp_input, transposed_input_ref);
  } else {
    transposed_input_ref = dummy_transposed_input.vref;
  }

  using PickShaderFn = vkapi::ShaderInfo (*)(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickWgFn = utils::uvec3 (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickLocalWgFn = utils::uvec3 (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const utils::uvec3&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);

  PickShaderFn pick_shader = nullptr;
  PickWgFn pick_global = nullptr;
  PickLocalWgFn pick_local = nullptr;

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (kind) {
    case TestKernelKind::GEMM_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMM_W_4X8>;
      pick_global = pick_q4gsw_linear_gemm_global_wg;
      pick_local = pick_q4gsw_linear_gemm_local_wg;
      break;
    case TestKernelKind::GEMV_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMV_W_4X8>;
      pick_global = pick_q4gsw_legacy_gemv_global_wg;
      pick_local = pick_q4gsw_legacy_gemv_local_wg;
      break;
    case TestKernelKind::GEMM_TIN_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMM_TIN_W_4X8>;
      pick_global = pick_q4gsw_linear_tin_gemm_global_wg;
      pick_local = pick_q4gsw_linear_tin_gemm_local_wg;
      break;
    case TestKernelKind::GEMV_W_4X8_NOSG:
      pick_shader = pick_forced_shader<TestKernelKind::GEMV_W_4X8_NOSG>;
      pick_global = pick_q4gsw_legacy_gemv_global_wg;
      pick_local = pick_q4gsw_legacy_gemv_local_wg;
      break;
    case TestKernelKind::PROD:
    default:
      VK_THROW("PROD kind must be dispatched via production entry points");
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader,
      pick_global,
      pick_local,
      {{output, vkapi::kWrite},
       {{fp_input,
         transposed_input_ref,
         packed_weight,
         packed_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      {apply_bias, K_val, static_cast<uint32_t>(group_size_val)},
      {weight_data, fp_input},
      resize_q4gsw_linear_node));
}

void add_fpa_q4gsw_linear_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    int32_t impl_selector_int,
    bool is_gemv,
    const ValueRef output) {
  TestKernelKind kind = selector_to_kind(impl_selector_int, is_gemv);

  if (kind == TestKernelKind::PROD) {
    // PROD: dispatch through the registered production op so the test exercises
    // the same wrapping the partitioner-emitted graph would hit.
    std::vector<ValueRef> q4gsw_linear_args = {
        fp_input,
        weight_data,
        weight_scales_data,
        group_size_ref,
        bias_data,
        output};
    VK_GET_OP_FN("et_vk.q4gsw_linear.default")(graph, q4gsw_linear_args);
    return;
  }

  if (kind == TestKernelKind::LEGACY) {
    // LEGACY: dispatch the legacy q4gsw linear shaders
    // (linear_q4gsw_tiled_* / linear_q4gsw_coop_*) directly via a private
    // copy of the dispatcher that was deleted from QuantizedLinear.cpp by the
    // W_4X8 commit. Uses pack_q4_linear_weight prepack and picks GEMM vs GEMV
    // internally based on input M.
    add_legacy_q4gsw_linear_node(
        graph,
        fp_input,
        weight_data,
        weight_scales_data,
        group_size_ref,
        bias_data,
        output);
    return;
  }

  if (kind == TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER ||
      kind == TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G1W64 ||
      kind == TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16 ||
      kind == TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8) {
    // Coop GEMV nc-Buffer variants — `kind` pins the (NUM_GROUPS,
    // WORKERS_PER_GROUP) reduction decomposition (g1w64 / g4w16 / g8w8). Weight
    // binding is the production nc-buffer SSBO (shared prepack with prefill).
    add_q4gsw_linear_coop_kc_forced_node(
        graph,
        fp_input,
        weight_data,
        weight_scales_data,
        group_size_ref,
        bias_data,
        output,
        kind);
    return;
  }

  add_q4gsw_linear_forced_node(
      graph,
      fp_input,
      weight_data,
      weight_scales_data,
      group_size_ref,
      bias_data,
      output,
      kind);
}

} // namespace

void test_fpa_q4gsw_linear_gemm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size_ref = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef impl_selector_ref = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int32_t impl_selector_int =
      graph.extract_scalar<int32_t>(impl_selector_ref);

  add_fpa_q4gsw_linear_node(
      graph,
      fp_input,
      weight_data,
      weight_scales_data,
      group_size_ref,
      bias_data,
      impl_selector_int,
      /*is_gemv=*/false,
      output);
}

void test_fpa_q4gsw_linear_gemv(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size_ref = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef impl_selector_ref = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int32_t impl_selector_int =
      graph.extract_scalar<int32_t>(impl_selector_ref);

  add_fpa_q4gsw_linear_node(
      graph,
      fp_input,
      weight_data,
      weight_scales_data,
      group_size_ref,
      bias_data,
      impl_selector_int,
      /*is_gemv=*/true,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      test_etvk.test_fpa_q4gsw_linear.gemm, test_fpa_q4gsw_linear_gemm);
  VK_REGISTER_OP(
      test_etvk.test_fpa_q4gsw_linear.gemv, test_fpa_q4gsw_linear_gemv);
}

} // namespace vkcompute
