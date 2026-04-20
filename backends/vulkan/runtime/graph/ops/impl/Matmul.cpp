/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Linear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Forward declaration
void resize_matmul_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// ── Cooperative matrix tile configuration (must match matmul_coopmat.glsl) ──

static constexpr uint32_t kCoopMatTileM = 64;
static constexpr uint32_t kCoopMatTileN = 64;
static constexpr uint32_t kCoopMatInvocations = 256; // 4 subgroups × 64

vkapi::ShaderInfo pick_matmul_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = "matmul_coopmat";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_matmul_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, kCoopMatTileN);
  uint32_t num_tiles_m = utils::div_up(M, kCoopMatTileM);
  return {num_tiles_n * kCoopMatInvocations, num_tiles_m, 1};
}

utils::uvec3 pick_matmul_coopmat_local_wg_size(
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
  return {kCoopMatInvocations, 1, 1};
}

void add_matmul_coopmat_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat2) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "matmul_coopmat requires buffer storage");

  ValueRef has_bias_ref = graph.add_scalar(false);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_coopmat_shader,
      pick_matmul_coopmat_global_wg_size,
      pick_matmul_coopmat_local_wg_size,
      // Inputs and Outputs — same binding order as matmul_vec
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers — same UBOs as matmul_vec
      {graph.sizes_ubo(mat1), graph.sizes_ubo(mat2)},
      // Push Constants
      {},
      // Specialization Constants (tile config hardcoded in shader)
      {},
      // Resize Args
      {has_bias_ref},
      // Resizing Logic
      resize_matmul_tiled_node));
}

// ── End cooperative matrix section ──

void resize_matmul_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  std::vector<int64_t> new_out_sizes(mat1_sizes);
  new_out_sizes.at(new_out_sizes.size() - 1) = mat2_sizes.back();
  new_out_sizes.at(new_out_sizes.size() - 2) =
      mat1_sizes.at(mat1_sizes.size() - 2);

  graph->virtual_resize(out, new_out_sizes);
}

// Minimum number of thread groups to target for good GPU occupancy. When the
// default 4-row tiling produces fewer threads than this, a smaller tile is
// selected to increase parallelism.
static constexpr uint32_t kMinOccupancyThreads = 4096;

// Returns the M tile size (1, 2, or 4) to use for the matmul shader. The
// largest tile that produces at least kMinOccupancyThreads thread groups is
// chosen; if even tile_m=1 doesn't meet the threshold, tile_m=1 is used.
uint32_t pick_matmul_tile_m(ComputeGraph* graph, const ValueRef out) {
  uint32_t N = graph->size_at<uint32_t>(-1, out);
  uint32_t M = graph->size_at<uint32_t>(-2, out);
  uint32_t B = graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;
  uint32_t n_groups = utils::div_up_4(N);
  // Try tile_m = 4, 2, 1 in descending order; pick the first that gives
  // enough threads.
  for (uint32_t tile_m : {4u, 2u, 1u}) {
    uint32_t total = n_groups * utils::div_up(M, tile_m) * B;
    if (total >= kMinOccupancyThreads) {
      return tile_m;
    }
  }
  return 1u;
}

vkapi::ShaderInfo pick_matmul_tiled_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  bool has_bias = graph->get_bool(resize_args.at(0));
  uint32_t tile_m = pick_matmul_tile_m(graph, out);

  bool is_buffer = graph->storage_type_of(out) == utils::kBuffer;
  // Use vec4 shader when all tensor widths are aligned to 4, even for buffers
  uint32_t K = graph->size_at<uint32_t>(-1, mat1);
  uint32_t N = graph->size_at<uint32_t>(-1, out);
  bool use_scalar = is_buffer && (K % 4 != 0 || N % 4 != 0);
  std::string base = use_scalar ? "matmul" : "matmul_vec";

  std::string kernel_name;
  if (has_bias) {
    kernel_name = tile_m <= 1 ? base + "_bias_tile_row_1"
        : tile_m <= 2         ? base + "_bias_tile_row_2"
                              : base + "_bias";
  } else {
    kernel_name = tile_m <= 1 ? base + "_tile_row_1"
        : tile_m <= 2         ? base + "_tile_row_2"
                              : base;
  }
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_matmul_tiled_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  uint32_t N = graph->size_at<uint32_t>(-1, out);
  uint32_t M = graph->size_at<uint32_t>(-2, out);
  uint32_t B = graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;
  uint32_t tile_m = pick_matmul_tile_m(graph, out);
  return {utils::div_up_4(N), utils::div_up(M, tile_m), B};
}

void add_matmul_tiled_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat2) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  ValueRef has_bias_ref = graph.add_scalar(false);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_tiled_shader,
      pick_matmul_tiled_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(mat1), graph.sizes_ubo(mat2)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {has_bias_ref},
      // Resizing Logic
      resize_matmul_tiled_node));
}

struct MatmulBiasParams final {
  float alpha;
  float beta;
};

void add_addmm_tiled_node(
    ComputeGraph& graph,
    const ValueRef bias,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out,
    float alpha_val,
    float beta_val) {
  VK_CHECK_COND(graph.packed_dim_of(bias) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat2) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);

  MatmulBiasParams params{alpha_val, beta_val};

  ValueRef has_bias_ref = graph.add_scalar(true);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_tiled_shader,
      pick_matmul_tiled_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2, bias}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(mat1), graph.sizes_ubo(mat2), graph.sizes_ubo(bias)},
      // Push Constants
      {PushConstantDataInfo(&params, sizeof(params))},
      // Specialization Constants
      {},
      // Resize Args
      {has_bias_ref},
      // Resizing Logic
      resize_matmul_tiled_node));
}

void matmul_tiled(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args: mat1, mat2, out
  ValueRef mat1 = args[0];
  ValueRef mat2 = args[1];
  ValueRef out = args[2];

  if (graph.val_is_tref(mat2)) {
    auto mat2_sizes = graph.sizes_of(mat2);
    int64_t B = mat2_sizes.size() >= 3 ? mat2_sizes.at(0) : 1;
    bool use_coopmat =
        graph.context()->adapter_ptr()->supports_cooperative_matrix() &&
        graph.storage_type_of(out) == utils::kBuffer;
    ValueRef packed = prepack_fp_linear_weight(
        graph, mat2, /*is_transposed=*/false, B,
        /*force_buffer=*/use_coopmat);
    if (use_coopmat) {
      add_linear_coopmat_node(
          graph, mat1, packed, kDummyValueRef, false, out,
          utils::safe_downcast<int32_t>(B));
    } else {
      add_linear_tiled_node(
          graph,
          mat1,
          packed,
          kDummyValueRef,
          false,
          out,
          utils::safe_downcast<int32_t>(B));
    }
  } else if (
      graph.context()->adapter_ptr()->supports_cooperative_matrix() &&
      graph.storage_type_of(out) == utils::kBuffer) {
    add_matmul_coopmat_node(graph, mat1, mat2, out);
  } else {
    add_matmul_tiled_node(graph, mat1, mat2, out);
  }
}

void addmm_tiled(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args: self, mat1, mat2, beta, alpha, out
  ValueRef self = args[0];
  ValueRef mat1 = args[1];
  ValueRef mat2 = args[2];
  ValueRef beta_ref = args[3];
  ValueRef alpha_ref = args[4];
  ValueRef out = args[5];

  float alpha_val = alpha_ref != kDummyValueRef
      ? graph.extract_scalar<float>(alpha_ref)
      : 1.0f;
  float beta_val =
      beta_ref != kDummyValueRef ? graph.extract_scalar<float>(beta_ref) : 1.0f;

  if (graph.val_is_tref(mat2)) {
    auto mat2_sizes = graph.sizes_of(mat2);
    int64_t B = mat2_sizes.size() >= 3 ? mat2_sizes.at(0) : 1;
    ValueRef packed =
        prepack_fp_linear_weight(graph, mat2, /*is_transposed=*/false, B);

    ValueRef packed_bias = kDummyValueRef;
    bool has_bias = graph.val_is_not_none(self);
    if (has_bias) {
      packed_bias = prepack_standard(
          graph,
          self,
          graph.storage_type_of(out),
          utils::kWidthPacked,
          /*passthrough=*/true);
    }
    add_linear_tiled_node(
        graph,
        mat1,
        packed,
        packed_bias,
        has_bias,
        out,
        utils::safe_downcast<int32_t>(B),
        alpha_val,
        beta_val);
  } else {
    ValueRef bias = prepack_standard(
        graph,
        self,
        graph.storage_type_of(out),
        utils::kWidthPacked,
        /*passthrough=*/true);
    add_addmm_tiled_node(graph, bias, mat1, mat2, out, alpha_val, beta_val);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.mm.default, matmul_tiled);
  VK_REGISTER_OP(aten.bmm.default, matmul_tiled);
  VK_REGISTER_OP(aten.addmm.default, addmm_tiled);
}

} // namespace vkcompute
