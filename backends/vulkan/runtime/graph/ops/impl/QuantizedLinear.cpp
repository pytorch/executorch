/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

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

// Per-shader coopmat tile geometry (must match each shader's yaml). The
// shaders restructured to the double-buffered reference (coopmat_mm_ref)
// use larger tiles and K-step 16; the rest keep the GemmCoopmat.h 64x64x32
// geometry. All use 256-thread workgroups.
//   linear_q4gsw_coopmat       128x128x16, 8 subgroups x 32 (forced)
//   linear_q8csw_coopmat       128x128x16, 8 subgroups x 32 (forced)
//   linear_dq8ca_q4gsw_coopmat 128x64x32,  4 subgroups x 64
//   linear_dq8ca_q8csw_coopmat 128x64x32,  4 subgroups x 64
// (The int8-MMA shaders stay on wave64: int8 WMMA at forced subgroup 32
// crashes the Xclipse PAL compiler.)
struct CoopmatTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
};
constexpr CoopmatTileDims kQ4gswCoopmatDims = {128, 128, 16};
constexpr CoopmatTileDims kQ8cswCoopmatDims = {128, 128, 16};
constexpr CoopmatTileDims kDq8caQ4gswCoopmatDims = {128, 64, 32};
constexpr CoopmatTileDims kDq8caQ8cswCoopmatDims = {128, 64, 32};

static CoopmatTileDims coopmat_tile_dims(const std::string& kernel_name) {
  // Exact prefix matches (the "linear_dq8ca_*" names must not match the
  // weight-only entries).
  if (kernel_name.rfind("linear_q4gsw_coopmat", 0) == 0) {
    return kQ4gswCoopmatDims;
  }
  if (kernel_name.rfind("linear_q8csw_coopmat", 0) == 0) {
    return kQ8cswCoopmatDims;
  }
  if (kernel_name.rfind("linear_dq8ca_q4gsw_coopmat", 0) == 0) {
    return kDq8caQ4gswCoopmatDims;
  }
  if (kernel_name.rfind("linear_dq8ca_q8csw_coopmat", 0) == 0) {
    return kDq8caQ8cswCoopmatDims;
  }
  return {kCoopmatTileM, kCoopmatTileN, kCoopmatTileK};
}

utils::uvec3 quantized_linear_global_wg_size(
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

  // Coopmat variants dispatch a 256-thread WG per 64x64 output tile.  Mirrors
  // GemmCoopmat.cpp's pick_linear_coopmat_global_wg_size — the multiplication
  // by kCoopmatInvocations cancels the framework's div_up, since
  // local_wg = {256, 1, 1}.
  if (shader.kernel_name.find("_coopmat") != std::string::npos) {
    const CoopmatTileDims dims = coopmat_tile_dims(shader.kernel_name);
    const uint32_t num_tiles_n = utils::div_up(N, dims.n);
    const uint32_t num_tiles_m = utils::div_up(M, dims.m);
    return {num_tiles_n * kCoopmatInvocations, num_tiles_m, 1};
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
  return {num_N_tiles, num_M_tiles, 1};
}

utils::uvec3 quantized_linear_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  // Coopmat variants use a 256-thread workgroup.
  if (shader.kernel_name.find("_coopmat") != std::string::npos) {
    return {kCoopmatInvocations, 1, 1};
  }

  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 1, 64};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

// Returns true when the q4gsw coopmat shader can be dispatched for this
// (M, N, K, dtype, output_storage, group_size) tuple. Preconditions match what
// linear_q4gsw_coopmat.glsl assumes; the subgroup_size == 64 check scopes this
// to wave64 devices (e.g. AMD RDNA), which the coopmat tiling is tuned for.
static bool can_use_q4gsw_coopmat(
    ComputeGraph* graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias,
    int64_t tile_m = kCoopmatTileM,
    int64_t tile_n = kCoopmatTileN,
    int64_t tile_k = kCoopmatTileK) {
  // Benchmark toggle: force the tiled fallback so a buffer PTE can serve as the
  // apples-to-apples baseline without re-exporting (see ET_VK_DISABLE_COOPMAT).
  if (std::getenv("ET_VK_DISABLE_COOPMAT") != nullptr) {
    return false;
  }
  // The coopmat shaders only build HAS_BIAS=false variants, so they would
  // silently drop a bias. Fall back to the tiled path (which applies bias at
  // runtime via the apply_bias spec constant) whenever a bias is present.
  if (!graph->val_is_none(bias)) {
    return false;
  }
  const auto* adapter = graph->context()->adapter_ptr();
  if (!adapter->supports_cooperative_matrix()) {
    return false;
  }
  if (adapter->subgroup_size() != 64) {
    return false;
  }
  if (graph->storage_type_of(output) != utils::kBuffer) {
    return false;
  }
  if (graph->dtype_of(output) != vkapi::kHalf) {
    return false;
  }

  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  const int64_t N = utils::val_at(-1, out_sizes);
  const int64_t M = utils::val_at(-2, out_sizes);
  const std::vector<int64_t> in_sizes = graph->sizes_of(fp_input);
  const int64_t K = utils::val_at(-1, in_sizes);

  if (M % tile_m != 0) {
    return false;
  }
  if (N % tile_n != 0) {
    return false;
  }
  if (K % tile_k != 0) {
    return false;
  }
  if (group_size % tile_k != 0) {
    return false;
  }
  return true;
}

vkapi::ShaderInfo pick_linear_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

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
            graph,
            output,
            fp_input,
            group_size,
            resize_args.at(2),
            kQ4gswCoopmatDims.m,
            kQ4gswCoopmatDims.n,
            kQ4gswCoopmatDims.k)) {
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

  // 8-bit per-channel weight-only (q8csw) coopmat. group_size doesn't apply,
  // so K is passed (it satisfies group_size % kCoopmatTileK == 0 when K does).
  // KNOWN ISSUE: shares the unresolved j>0 N-subtile correctness bug with the
  // other coopmat shaders (under investigation via the consolidated bench).
  if (!weight_is_4bit && !is_gemv_case) {
    const int64_t K = graph->size_at<int64_t>(-1, fp_input);
    if (can_use_q4gsw_coopmat(
            graph,
            output,
            fp_input,
            K,
            resize_args.at(2),
            kQ8cswCoopmatDims.m,
            kQ8cswCoopmatDims.n,
            kQ8cswCoopmatDims.k)) {
      std::string kernel_name = "linear_q8csw_coopmat";
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
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef int_input = args.at(1).refs.at(1);
  (void)int_input;
  const ValueRef int_weight = args.at(1).refs.at(5);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat<int8> shader for 4-bit dq8ca dispatches when the device
  // exposes INT8 coopmat properties and the shape aligns; tiled otherwise.
  if (weight_is_4bit && !is_gemv_case) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    if (can_use_q4gsw_coopmat(
            graph,
            out,
            fp_input,
            group_size,
            resize_args.at(2),
            kDq8caQ4gswCoopmatDims.m,
            kDq8caQ4gswCoopmatDims.n,
            kDq8caQ4gswCoopmatDims.k)) {
      std::string kernel_name = "linear_dq8ca_q4gsw_coopmat";
      add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
      add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(out));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  // Use the coopmat<int8> shader for 8-bit per-channel dq8ca. Same matrix-unit
  // path and shape/dtype preconditions; group_size doesn't apply for
  // per-channel weights, so K is passed (it always satisfies
  // group_size % kCoopmatTileK == 0 when K does).
  if (!weight_is_4bit && !is_gemv_case) {
    const int64_t K = graph->size_at<int64_t>(-1, fp_input);
    if (can_use_q4gsw_coopmat(
            graph,
            out,
            fp_input,
            K,
            resize_args.at(2),
            kDq8caQ8cswCoopmatDims.m,
            kDq8caQ8cswCoopmatDims.n,
            kDq8caQ8cswCoopmatDims.k)) {
      std::string kernel_name = "linear_dq8ca_q8csw_coopmat";
      add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
      add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(out));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_";
  if (weight_is_4bit) {
    kernel_name += "dq8ca_q4gsw";
  } else {
    kernel_name += "dq8ca_q8csw";
  }

  if (weight_is_4bit && is_gemv_case) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(kernel_name);
}

//
// Prepacking nodes
//

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef qmat2_data) {
  VK_CHECK_COND(
      weight_quant_config.nbits == 8 || weight_quant_config.nbits == 4);

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

  std::string kernel_name = weight_quant_config.nbits == 4
      ? "pack_q4_linear_weight"
      : "pack_q8_linear_weight";
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

  utils::uvec3 global_wg_size;
  if (weight_quant_config.nbits == 4) {
    // For 4-bit quantization, each thread writes out two adjacent blocks
    global_wg_size = {
        utils::safe_downcast<uint32_t>(utils::div_up(num_blocks_K, int64_t(2))),
        utils::safe_downcast<uint32_t>(num_blocks_N),
        1u};
  } else {
    global_wg_size = {
        utils::safe_downcast<uint32_t>(num_blocks_N),
        utils::safe_downcast<uint32_t>(num_blocks_K),
        1u};
  }

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
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
  // 3rd coopmat spec const: num_groups (4-bit q4gsw) or K-chunk count (8-bit
  // q8csw). Either way it is the trip count of the coopmat loop, passed as a
  // spec constant to avoid the Xclipse driver crash on UBO-derived bounds.
  int32_t num_groups = 0;
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    num_groups = graph.size_at<int32_t>(-1, fp_input) / group_size_val;
  } else {
    num_groups = graph.size_at<int32_t>(-1, fp_input) /
        static_cast<int32_t>(kQ8cswCoopmatDims.k);
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_qw_shader,
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
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
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
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

  // Allow per-channel symmetric INT8 weight alongside the original
  // per-group INT4. Both flows reuse the same dq8ca packed-int8 input
  // tile + integer accumulator; the shader picks the right inner loop
  // based on the dispatched kernel name.
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(
      (weight_quant_config.granularity == kPerGroup &&
       weight_quant_config.nbits == 4) ||
      (weight_quant_config.granularity == kPerChannel &&
       weight_quant_config.nbits == 8));

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
  } else {
    coopmat_k_iters = K_dim / static_cast<int32_t>(kDq8caQ8cswCoopmatDims.k);
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_dqa_qw_shader,
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
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
      // 4th spec const: output width N for coopMatStore (see add_linear_qw_node).
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

  // Otherwise, input is dynamically quantized. Supports either per-group
  // 4-bit or per-channel 8-bit symmetric weights (both reuse the same
  // dq8ca path, but with different shaders dispatched downstream).
  VK_CHECK_COND(
      weight_quant_config.nbits == 4 || weight_quant_config.nbits == 8);

  int64_t num_groups = 1;
  if (weight_quant_config.granularity == kPerGroup) {
    num_groups = graph.size_at<int64_t>(-2, weight_scales_data);
  }

  TmpTensor int_input_sums(
      &graph,
      {num_groups, K},
      graph.dtype_of(output),
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

void linear_q4gsw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

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
      group_size, // group size
      bias_data,
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

void linear_dq8ca_q8csw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // W8A8 dynamic: per-channel symmetric INT8 weights + per-token dynamic
  // INT8 activations. No group_size — per-channel weight quant has no
  // groups. We piggyback on the existing dq8ca pipeline by treating
  // per-channel as a single group covering the whole K dim, so the
  // quantize_and_pack_4h4w_with_group_sums helper degenerates to a
  // single-group sum (which the q8csw shader ignores anyway, since the
  // epilog uses (acc - input_zp * weight_sum) per-row instead).
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  QuantizationConfig weight_quant_config(8, kPerChannel, {});

  // Synthesize group_size = K so num_groups = 1 in the existing flow.
  const int64_t K = graph.size_at<int64_t>(-1, fp_input);
  const ValueRef group_size_ref = graph.add_scalar<int64_t>(K);

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
      group_size_ref,
      bias_data,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.linear_q8csw.default, linear_q8csw);
  VK_REGISTER_OP(et_vk.linear_q4gsw.default, linear_q4gsw);
  VK_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw);
  VK_REGISTER_OP(et_vk.linear_dq8ca_q8csw.default, linear_dq8ca_q8csw);
}

} // namespace vkcompute
