/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <limits>

namespace vkcompute {

//
// Shader dispatch utilities
//

void resize_linear_qw_node(
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

// Per-shader coopmat tile geometry (must match each shader's yaml).
// Workgroup size (wg_size) = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE.
//   linear_q4gsw_coopmat       128x64x32, 2x2 x 32 -> 128
//   linear_dq8ca_q4gsw_coopmat 128x64x32, 2x2 x 64 or 4x2 x 32 -> 256
struct CoopmatTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  // Threads per workgroup = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE. MUST match
  // the WG_SIZE the shader yaml resolves to, or the launched thread count won't
  // match the shader's staging passes (out-of-bounds).
  uint32_t wg_size;
};
// linear_qw_coopmat.yaml: 128x64, 2x2 subgroup grid, sg32 -> WG_SIZE 128.
constexpr CoopmatTileDims kQ4gswCoopmatDims = {128, 64, 32, 128};
// linear_dq8ca_qw_coopmat.yaml: 128x64, sg64 or sg32 -> WG_SIZE 256.
constexpr CoopmatTileDims kDq8caQ4gswCoopmatDims = {128, 64, 32, 256};

// Static Workgroup storage declared by the bias-free shader variants. Keep
// these formulas in sync with the shared arrays in the corresponding GLSL.
constexpr uint32_t padded_fp16_stride_uvec4(uint32_t elements) {
  constexpr uint32_t kFp16PerUvec4 = 8u;
  // The GLSL deliberately adds one uvec4 of skew, even for aligned rows.
  return (elements + kFp16PerUvec4) / kFp16PerUvec4;
}

constexpr uint32_t q4gsw_coopmat_shared_memory_bytes(
    const CoopmatTileDims& dims) {
  // Two ping-pong arrays: Ash[M][padded K] and Bsh[K][padded N].
  return 2u *
      (dims.m * padded_fp16_stride_uvec4(dims.k) +
       dims.k * padded_fp16_stride_uvec4(dims.n)) *
      16u;
}

constexpr uint32_t kQ4gswCoopmatSharedMemoryBytes =
    q4gsw_coopmat_shared_memory_bytes(kQ4gswCoopmatDims);

constexpr uint32_t dq8ca_q4gsw_coopmat_shared_memory_bytes(uint32_t mma_k) {
  constexpr uint32_t kScalarBytes = 4u;
  const uint32_t num_k_slabs = kDq8caQ4gswCoopmatDims.k / mma_k;
  // A is byte-packed; B is uint-packed with one skew uint per column.
  const uint32_t a_double_buffer_bytes =
      2u * kDq8caQ4gswCoopmatDims.m * kDq8caQ4gswCoopmatDims.k;
  const uint32_t b_double_buffer_bytes = 2u * num_k_slabs *
      kDq8caQ4gswCoopmatDims.n * (mma_k / 4u + 1u) * kScalarBytes;
  const uint32_t activation_params_bytes =
      kDq8caQ4gswCoopmatDims.m * 2u * kScalarBytes;
  const uint32_t weight_params_bytes =
      2u * kDq8caQ4gswCoopmatDims.n * 2u * kScalarBytes;
  return a_double_buffer_bytes + b_double_buffer_bytes +
      activation_params_bytes + weight_params_bytes;
}

constexpr uint32_t kDq8caQ4gswCoopmatSg32SharedMemoryBytes =
    dq8ca_q4gsw_coopmat_shared_memory_bytes(32u);
constexpr uint32_t kDq8caQ4gswCoopmatSg64SharedMemoryBytes =
    dq8ca_q4gsw_coopmat_shared_memory_bytes(16u);

static_assert(kQ4gswCoopmatSharedMemoryBytes == 29696u);
static_assert(kDq8caQ4gswCoopmatSg32SharedMemoryBytes == 14848u);
static_assert(kDq8caQ4gswCoopmatSg64SharedMemoryBytes == 15360u);

static CoopmatTileDims coopmat_tile_dims(const std::string& kernel_name) {
  // Exact prefix matches (the "linear_dq8ca_*" names must not match the
  // weight-only entries).
  if (kernel_name.rfind("linear_q4gsw_coopmat", 0) == 0) {
    return kQ4gswCoopmatDims;
  }
  if (kernel_name.rfind("linear_dq8ca_q4gsw_coopmat", 0) == 0) {
    return kDq8caQ4gswCoopmatDims;
  }
  return {kCoopmatTileM, kCoopmatTileN, kCoopmatTileK, kCoopmatInvocations};
}

GlobalWorkGrid quantized_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  // width
  const uint32_t N = utils::val_at(-1, out_sizes);
  // height
  const uint32_t M = utils::val_at(-2, out_sizes);

  // Coopmat variants dispatch one workgroup per shader-specific output tile.
  // Scaling x by the workgroup size cancels the dispatcher's division by the
  // required local size.
  if (shader.kernel_name.find("_coopmat") != std::string::npos) {
    const CoopmatTileDims dims = coopmat_tile_dims(shader.kernel_name);
    const uint32_t num_tiles_n = utils::div_up(N, dims.n);
    const uint32_t num_tiles_m = utils::div_up(M, dims.m);
    return GlobalWorkGrid(
        {num_tiles_n * dims.wg_size, num_tiles_m, 1u},
        kTiledWorkGrid,
        LocalWorkGroup(dims.wg_size, 1u, 1u));
  }

  uint32_t N_per_tile = 4;
  uint32_t M_per_tile = 4;

  // For 4-bit weights, each output tile contains 8 columns
  if (shader.kernel_name.find("q4") != std::string::npos) {
    N_per_tile = 8;
  }
  if (shader.kernel_name.find("coop") != std::string::npos) {
    M_per_tile = 1;
  }

  if (shader.kernel_name.find("q8ta_q8csw_tiled") != std::string::npos) {
    N_per_tile = 8;
  }

  const uint32_t num_N_tiles = utils::div_up(N, N_per_tile);
  const uint32_t num_M_tiles = utils::div_up(M, M_per_tile);

  // Otherwise, each output tile contains 4 columns and 4 rows
  if (shader.kernel_name.find("_coop") != std::string::npos) {
    return GlobalWorkGrid(
        {num_N_tiles, num_M_tiles, 1u},
        kTiledWorkGrid,
        LocalWorkGroup(1u, 1u, 64u));
  }
  return GlobalWorkGrid({num_N_tiles, num_M_tiles, 1u}, kTiledWorkGrid);
}

LocalWorkGroup quantized_linear_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (gwg.required_lwg_size().is_valid()) {
    return pick_required_lwg(graph, shader, gwg, args, resize_args);
  }
  return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
}

static bool has_q4gsw_coopmat_compatible_layout(
    ComputeGraph* graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias,
    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k,
    uint32_t wg_size) {
  // The coopmat shaders only build HAS_BIAS=false variants, so they would
  // silently drop a bias. Fall back to the tiled path (which applies bias at
  // runtime via the apply_bias spec constant) whenever a bias is present.
  if (!graph->val_is_none(bias)) {
    return false;
  }
  // Coopmat shaders use flat, width-packed buffer addressing. A singleton
  // batch dimension has the same layout and is how LLM prefill reaches this
  // operator after the decomposed linear pattern is fused.
  const int64_t output_dim = graph->dim_of(output);
  const int64_t input_dim = graph->dim_of(fp_input);
  if (output_dim != input_dim || (output_dim != 2 && output_dim != 3)) {
    return false;
  }
  if (graph->storage_type_of(output) != utils::kBuffer ||
      graph->storage_type_of(fp_input) != utils::kBuffer) {
    return false;
  }
  if (graph->dtype_of(output) != vkapi::kHalf ||
      graph->dtype_of(fp_input) != vkapi::kHalf) {
    return false;
  }
  if (graph->packed_dim_of(output) != WHCN::kWidthDim ||
      graph->packed_dim_of(fp_input) != WHCN::kWidthDim ||
      !graph->has_standard_axis_map(output) ||
      !graph->has_standard_axis_map(fp_input)) {
    return false;
  }

  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  const int64_t N = utils::val_at(-1, out_sizes);
  const int64_t M = utils::val_at(-2, out_sizes);
  const std::vector<int64_t> in_sizes = graph->sizes_of(fp_input);
  const int64_t input_M = utils::val_at(-2, in_sizes);
  const int64_t K = utils::val_at(-1, in_sizes);

  if (output_dim == 3 && (out_sizes.at(0) != 1 || in_sizes.at(0) != 1)) {
    return false;
  }
  if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0 || M != input_M) {
    return false;
  }
  // The shaders have no edge guards, and quantization groups must contain
  // whole K tiles.
  if (M % tile_m != 0 || N % tile_n != 0 || K % tile_k != 0 ||
      group_size % tile_k != 0 || K % group_size != 0) {
    return false;
  }

  const uint64_t num_tiles_n = static_cast<uint64_t>(N / tile_n);
  const uint64_t num_tiles_m = static_cast<uint64_t>(M / tile_m);
  const utils::uvec3 max_wg_count =
      graph->context()->adapter_ptr()->max_compute_workgroup_count();
  // GWG.x stores tile_count * WG_SIZE before dispatch divides by WG_SIZE.
  if (num_tiles_n > max_wg_count[0] || num_tiles_m > max_wg_count[1] ||
      max_wg_count[2] == 0u ||
      num_tiles_n * wg_size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  return true;
}

static bool can_request_subgroup_size(
    const vkapi::Adapter* adapter,
    uint32_t subgroup_size) {
  return adapter->supports_required_subgroup_size_for_compute() &&
      subgroup_size >= adapter->min_subgroup_size() &&
      subgroup_size <= adapter->max_subgroup_size();
}

static bool has_q4gsw_coopmat_device_resources(
    vkapi::Adapter* adapter,
    uint32_t subgroup_size,
    uint32_t wg_size,
    uint32_t shared_memory_bytes) {
  const utils::uvec3 max_wg_size = adapter->max_compute_workgroup_size();
  // SUBGROUP_SIZE in the shader YAML becomes a required subgroup size on the
  // Vulkan pipeline, so both its range and subgroup-count limit apply.
  return adapter->supports_cooperative_matrix() &&
      adapter->supports_vulkan_memory_model() &&
      adapter->supports_float16_shader_types() &&
      adapter->supports_16bit_storage_buffers() &&
      adapter->supports_subgroup_compute_basic() &&
      can_request_subgroup_size(adapter, subgroup_size) &&
      wg_size % subgroup_size == 0u &&
      wg_size / subgroup_size <= adapter->max_compute_workgroup_subgroups() &&
      wg_size <= adapter->max_compute_workgroup_invocations() &&
      wg_size <= max_wg_size[0] && max_wg_size[1] >= 1u &&
      max_wg_size[2] >= 1u &&
      shared_memory_bytes <= adapter->max_compute_shared_memory_size();
}

bool can_use_q4gsw_coopmat(
    ComputeGraph& graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias) {
  auto* adapter = graph.context()->adapter_ptr();
  constexpr uint32_t kRequiredSubgroupSize = 32u;
  return adapter->supports_fp16_cooperative_matrix(16, 16, 16) &&
      adapter->supports_fp16_cooperative_matrix_accumulator(16, 16) &&
      has_q4gsw_coopmat_device_resources(
             adapter,
             kRequiredSubgroupSize,
             kQ4gswCoopmatDims.wg_size,
             kQ4gswCoopmatSharedMemoryBytes) &&
      has_q4gsw_coopmat_compatible_layout(
             &graph,
             output,
             fp_input,
             group_size,
             bias,
             kQ4gswCoopmatDims.m,
             kQ4gswCoopmatDims.n,
             kQ4gswCoopmatDims.k,
             kQ4gswCoopmatDims.wg_size);
}

static const char* pick_dq8ca_q4gsw_coopmat_variant(
    ComputeGraph* graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias) {
  auto* adapter = graph->context()->adapter_ptr();
  // After the int8 MMA, the shader uses fp32 matrices for scale/correction
  // math and converts the result through an fp16 accumulator for the store.
  if (!has_q4gsw_coopmat_compatible_layout(
          graph,
          output,
          fp_input,
          group_size,
          bias,
          kDq8caQ4gswCoopmatDims.m,
          kDq8caQ4gswCoopmatDims.n,
          kDq8caQ4gswCoopmatDims.k,
          kDq8caQ4gswCoopmatDims.wg_size) ||
      !adapter->supports_int8_shader_types() ||
      !adapter->supports_fp16_cooperative_matrix_accumulator(16, 16) ||
      !adapter->supports_fp32_cooperative_matrix_accumulator(16, 16)) {
    return nullptr;
  }

  // Each variant is tuned for, and pinned to, the device's native width.
  const uint32_t subgroup_size = adapter->subgroup_size();
  if (subgroup_size == 32 &&
      has_q4gsw_coopmat_device_resources(
          adapter,
          32,
          kDq8caQ4gswCoopmatDims.wg_size,
          kDq8caQ4gswCoopmatSg32SharedMemoryBytes) &&
      adapter->supports_int8_cooperative_matrix(16, 16, 32)) {
    return "_sg32";
  }
  // Preserve the wave64/K16 geometry used by AMD/RDNA, but select it by
  // capabilities rather than by vendor name.
  if (subgroup_size == 64 &&
      has_q4gsw_coopmat_device_resources(
          adapter,
          64,
          kDq8caQ4gswCoopmatDims.wg_size,
          kDq8caQ4gswCoopmatSg64SharedMemoryBytes) &&
      adapter->supports_int8_cooperative_matrix(16, 16, 16)) {
    return "";
  }
  return nullptr;
}

vkapi::ShaderInfo pick_linear_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_int_weight = args.at(1).refs.at(1);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat shader for 4-bit, non-gemv, buffer-output, half-dtype
  // dispatches when shape alignment allows; tiled remains the fallback.
  if (weight_is_4bit && !is_gemv_case) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    if (can_use_q4gsw_coopmat(
            *graph, output, fp_input, group_size, resize_args.at(2))) {
      std::string kernel_name = "linear_q4gsw_coopmat";
      // Output storage is buffer (gated above); weight storage matches the
      // existing variants.
      add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
      add_storage_type_suffix(
          kernel_name, graph->storage_type_of(packed_int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(output));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_";
  if (weight_is_4bit) {
    kernel_name += "q4gsw";
  } else {
    kernel_name += "q8csw";
  }

  if (weight_is_4bit && is_gemv_case) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(
      kernel_name, graph->storage_type_of(packed_int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo pick_linear_dqa_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef int_input = args.at(1).refs.at(1);
  (void)int_input;
  const ValueRef input_zp = args.at(1).refs.at(4);
  const ValueRef int_weight = args.at(1).refs.at(5);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat<int8> shader when the device exposes the exact matrix
  // tuple used by its native subgroup size and the shape aligns.
  if (weight_is_4bit && !is_gemv_case) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    const char* variant = pick_dq8ca_q4gsw_coopmat_variant(
        graph, out, fp_input, group_size, resize_args.at(2));
    if (variant != nullptr) {
      std::string kernel_name = "linear_dq8ca_q4gsw_coopmat";
      kernel_name += variant;
      add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
      add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(out));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_dq8ca_q4gsw";
  kernel_name += is_gemv_case ? "_coop" : "_tiled";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  add_zp_dtype_mode_suffix(kernel_name, graph->dtype_of(input_zp));

  return VK_KERNEL_FROM_STR(kernel_name);
}

//
// Prepacking nodes
//

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef qmat2_data,
    const bool use_unsigned_dot) {
  VK_CHECK_COND(
      weight_quant_config.nbits == 8 || weight_quant_config.nbits == 4);
  VK_CHECK_COND(!use_unsigned_dot || weight_quant_config.nbits == 8);

  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  int64_t qmat2_width = qmat2_orig_sizes.at(ndim - 1);
  int64_t qmat2_height = qmat2_orig_sizes.at(ndim - 2);

  int64_t K;
  int64_t N;
  if (weight_quant_config.nbits == 4) {
    // For 4-bit quantization, weight source data has shape [N, K/2]. Each byte
    // contains 2 * 4-bit values.
    K = qmat2_width * 2;
    N = qmat2_height;
  } else {
    // For 8-bit quantization, the weight source data has shape [N, K]
    K = qmat2_width;
    N = qmat2_height;
  }

  // Sanity check that assumptions are correct. Data loads along the innermost
  // dimension must be well aligned along texel boundaries.
  if (weight_quant_config.nbits == 4) {
    VK_CHECK_COND(K % 8 == 0);
  } else {
    VK_CHECK_COND(K % 4 == 0);
  }

  // The packing format packs the weight tensor into blocks of 4 columns (K) and
  // 4 rows (N)
  int64_t N_per_block = 4;
  int64_t K_per_block = 4;

  // For 4 bit, quantization, the amount of information contained in one block
  // can be doubled. Each block will contain data for 8 rows (N) instead of the
  // usual 4.
  if (weight_quant_config.nbits == 4) {
    N_per_block = 8;
  }

  // To figure out the size of the output tensor, determine the number of blocks
  // along each dimension.
  const int64_t num_blocks_K = utils::div_up(K, K_per_block);
  const int64_t num_blocks_N = utils::div_up(N, N_per_block);

  // The blocks are arranged in a transposed manner, such that the transposed
  // weight block is indexed like packed_weights[k4][n4] - this is to allow for
  // optimal memory coalescing when computing GEMM.
  int64_t output_height = num_blocks_K;
  // The base dtype of the packed tensor is int32 (each int32 contains 4x 8bit
  // values) and each block is represented as a ivec4. Therefore the width dim
  // of the packed tensor is multiplied by 4.
  int64_t output_width = num_blocks_N * 4;

  // For 4 bit quantization, The blocks are arranged without the transposition,
  // such that a weight block is accessed like packed_weights[n8][k4]. This is
  // an optimization targeted for LLMs, which need to compute GEMV as well as
  // GEMM. This memory layout provides better performance for the co-operative
  // algorithm used to compute GEMV, at the cost of slightly reducing GEMM
  // performance.
  if (weight_quant_config.nbits == 4) {
    output_height = num_blocks_N;
    output_width = num_blocks_K * 4;
  }

  // Store the original sizes of the weight data to pass to the shader
  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  std::vector<int64_t> qmat2_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }
  std::string kernel_name;
  if (weight_quant_config.nbits == 4) {
    kernel_name = "pack_q4_linear_weight";
  } else {
    kernel_name = use_unsigned_dot ? "pack_q8_linear_weight_unsigned"
                                   : "pack_q8_linear_weight";
  }
  add_storage_type_suffix(kernel_name, storage_type);

  // Check prepack cache before creating a new prepack node. This avoids
  // allocating a duplicate output tensor when the same weight data has already
  // been prepacked with the same kernel (e.g. tied embedding/linear weights).
  ValueRef cached = graph.get_cached_prepack(qmat2_data, kernel_name);
  if (is_valid(cached)) {
    return cached;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kInt, storage_type, utils::kWidthPacked);

  utils::uvec3 global_extents;
  if (weight_quant_config.nbits == 4) {
    // For 4-bit quantization, each thread writes out two adjacent blocks
    global_extents = {
        utils::safe_downcast<uint32_t>(utils::div_up(num_blocks_K, int64_t(2))),
        utils::safe_downcast<uint32_t>(num_blocks_N),
        1u};
  } else {
    global_extents = {
        utils::safe_downcast<uint32_t>(num_blocks_N),
        utils::safe_downcast<uint32_t>(num_blocks_K),
        1u};
  }
  const GlobalWorkGrid gwg(global_extents, kTiledWorkGrid);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      gwg,
      graph.create_lwg(gwg),
      // Inputs and Outputs
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

//
// Dispatch nodes
//

/*
 * Shader dispatch for linear with quantized weight but fp activations.
 */
void add_linear_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_scales,
    const ValueRef packed_weight_zeros,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  // Only certain quantization types supported at the moment
  VK_CHECK_COND(
      weight_quant_config.granularity == kPerChannel ||
      weight_quant_config.granularity == kPerGroup);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(
      weight_quant_config.nbits == 8 || weight_quant_config.nbits == 4);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  int32_t K4_per_group = 0;
  // 3rd coopmat spec const: num_groups (trip count of the coopmat loop),
  // passed as a spec constant to avoid the Xclipse UBO-derived bounds crash.
  int32_t num_groups = 0;
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    num_groups = graph.size_at<int32_t>(-1, fp_input) / group_size_val;
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_qw_shader,
      quantized_linear_gwg,
      quantized_linear_lwg,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      // 4th spec const: output width N. The coopmat shaders must take N for
      // coopMatStore address math from a spec constant, not the sizes UBO
      // (Xclipse driver miscompiles UBO-derived store offsets/strides).
      {apply_bias,
       K4_per_group,
       num_groups,
       graph.size_at<int32_t>(-1, output)},
      // Resize args (resize_args.at(2) = bias_data, read by the coopmat gate)
      {is_4bit_flag, weight_data, bias_data},
      // Resizing Logic
      resize_linear_qw_node));
}

void add_linear_qa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  VK_CHECK_COND(input_quant_config.granularity == kPerTensor);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  float scale = graph.extract_scalar<float>(input_scale_data);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp_data);

  // Get shader for quantized linear
  std::string kernel_name = "linear_q8ta_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(packed_int_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_gwg,
      quantized_linear_lwg,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{packed_int_input,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {apply_bias},
      // Resize args
      {fp_input},
      // Resizing Logic
      nullptr));
}

void add_linear_dqa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  VK_CHECK_COND(input_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(input_quant_config.is_dynamic);

  VK_CHECK_COND(weight_quant_config.granularity == kPerGroup);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 4);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  int32_t K4_per_group = 0;
  int32_t coopmat_k_iters = 0;
  const int32_t K_dim = graph.size_at<int32_t>(-1, fp_input);
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    coopmat_k_iters = K_dim / group_size_val;
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_dqa_qw_shader,
      quantized_linear_gwg,
      quantized_linear_lwg,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input,
         packed_int_input,
         int_input_sums,
         packed_input_scale,
         packed_input_zp,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      // 4th spec const: output width N for coopMatStore (see
      // add_linear_qw_node).
      {apply_bias,
       K4_per_group,
       coopmat_k_iters,
       graph.size_at<int32_t>(-1, output)},
      // Resize args (resize_args.at(2) = bias_data, read by the coopmat gate)
      {is_4bit_flag, weight_data, bias_data},
      // Resizing Logic
      resize_linear_qw_node));
}

//
// High level operator impl
//

void quantized_linear_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef weight_zeros_data,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef output) {
  std::vector<int64_t> input_sizes = graph.sizes_of(fp_input);
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);

  const int64_t K = utils::val_at(-1, input_sizes);
  // K (input channels) must be a multiple of 4 to ensure that reading a group
  // of 4 input channels from the input tensor will be aligned on a texel
  // boundary.
  VK_CHECK_COND(K % 4 == 0);

  // Prepack weight data

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  // Weight affine quant not supported at the moment
  const ValueRef packed_weight_zeros = kDummyValueRef;

  // Prepack bias data

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shdaer variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  // Use weight only quantized linear if at least one is true:
  // 1. Device does not support int8 dot product
  // 2. Input is not quantized
  if (!graph.can_use_int8_dot_product() ||
      input_quant_config.granularity == kNoQuantization) {
    add_linear_qw_node(
        graph,
        weight_quant_config,
        fp_input,
        weight_data,
        packed_weight,
        packed_weight_scales,
        packed_weight_zeros,
        group_size,
        bias_data,
        packed_bias,
        output);

    return;
  }
  // Otherwise, use input and weight quantized linear computed with integer
  // accumulation

  // Input scale/zero point only used for activation & weight quantized linear
  ValueRef packed_input_scale = input_scale;
  ValueRef packed_input_zp = input_zp;
  if (graph.val_is_tref(input_scale)) {
    VK_CHECK_COND(graph.val_is_tref(packed_input_zp));
    packed_input_scale = prepack_standard(
        graph, input_scale, utils::kTexture3D, utils::kWidthPacked);
    packed_input_zp = prepack_standard(
        graph, input_zp, utils::kTexture3D, utils::kWidthPacked);
  }

  // Pre-computed per quant group weight sums are needed for int accumulation,
  // but not for weight only
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  // Allocate temporary tensor to store quantized and packed input
  TmpTensor packed_int_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4H4W);

  // Non dynamically quantized input case
  if (!input_quant_config.is_dynamic) {
    add_quantize_and_pack_4h4w_node(
        graph,
        input_quant_config,
        fp_input,
        packed_input_scale,
        packed_input_zp,
        input_scale,
        input_zp,
        packed_int_input,
        group_size);

    add_linear_qa_qw_node(
        graph,
        input_quant_config,
        weight_quant_config,
        fp_input,
        packed_int_input,
        packed_input_scale,
        packed_input_zp,
        input_scale,
        input_zp,
        weight_data,
        packed_weight,
        packed_weight_sums,
        packed_weight_scales,
        group_size,
        bias_data,
        packed_bias,
        output);

    return;
  }

  // Otherwise, input is dynamically quantized. Currently only per group 4-bit
  // quantized weights is supported for this mode.
  VK_CHECK_COND(weight_quant_config.nbits == 4);

  int64_t num_groups = 1;
  if (weight_quant_config.granularity == kPerGroup) {
    num_groups = graph.size_at<int64_t>(-2, weight_scales_data);
  }

  // Per-group int8 input sums buffer, indexed as ivec4[group_idx * M4 + m4]
  // by both the producer (quantize_and_pack_4h4w_with_group_sums.glsl) and the
  // consumer (linear_int8_input_sums_load.glslh). Capacity must therefore be
  // num_groups * M4 ivec4 texels, sized by the input row count M -- NOT K.
  // dtype is kInt to match the shaders' `int`/ivec4 binding (each texel is 4
  // int32 sums = 16 bytes).
  const int64_t M = utils::val_at(-2, input_sizes);
  const int64_t M4 = utils::div_up(M, int64_t(4));
  TmpTensor int_input_sums(
      &graph,
      {num_groups * M4 * 4},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  add_quantize_and_pack_4h4w_with_group_sums_node(
      graph,
      input_quant_config,
      fp_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      packed_int_input,
      group_size);

  add_linear_dqa_qw_node(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      packed_int_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      input_scale,
      input_zp,
      weight_data,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      group_size,
      bias_data,
      packed_bias,
      output);
}

void add_q4gsw_coopmat_linear_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef output) {
  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);
  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef,
      kDummyValueRef,
      weight_data,
      kDummyValueRef,
      weight_scales_data,
      kDummyValueRef,
      group_size,
      bias_data,
      output);
}

void linear_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(8, kPerTensor, {}, false);
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      kDummyValueRef, // group_size
      bias_data,
      output);
}

void linear_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      bias_data,
      output);
}

// aten._weight_int8pack_mm is what the AOT weight-only int8 fusion
// (FuseQuantizedOpsTransform) emits. It carries the same operands as
// et_vk.linear_q8csw minus the bias, so it runs through the same
// implementation.
void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      kDummyValueRef, // bias
      output);
}

void linear_dq8ca_q4gsw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      group_size, // group_size
      bias_data,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.linear_q8csw.default, linear_q8csw);
  VK_REGISTER_OP(aten._weight_int8pack_mm.default, weight_int8pack_mm);
  VK_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw);
}

} // namespace vkcompute
