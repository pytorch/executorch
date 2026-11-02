// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Preprocess.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/q4gsw/Q4gswLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

enum class TestKernelKind {
  PROD,
  GEMM_W_4X8,
  GEMM_TIN_W_4X8,
  GEMV_W_4X8,
  GEMV_W_4X8_NOSG,
  LEGACY,
  GEMV_COOP_W_4X8_NC_BUFFER,

  GEMV_COOP_W_4X8_NC_BUFFER_G1W64,
  GEMV_COOP_W_4X8_NC_BUFFER_G4W16,
  GEMV_COOP_W_4X8_NC_BUFFER_G8W8,
};

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

template <uint32_t NUM_GROUPS, uint32_t WORKERS_PER_GROUP>
GlobalWorkGrid pick_q4gsw_coop_gwg(
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
  const LocalWorkGroup lwg(1u, NUM_GROUPS, WORKERS_PER_GROUP);
  GlobalWorkGrid gwg({wgs_along_x * NUM_GROUPS, 1u, 1u}, kLinearWorkGrid);
  gwg.wrap_linear_dispatch(
      graph->context()->adapter_ptr()->max_compute_workgroup_count(), lwg);
  return gwg;
}

template <uint32_t NUM_GROUPS, uint32_t WORKERS_PER_GROUP>
LocalWorkGroup pick_q4gsw_coop_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)gwg;
  (void)args;
  (void)resize_args;
  return LocalWorkGroup(1u, NUM_GROUPS, WORKERS_PER_GROUP);
}

constexpr uint32_t kGemvSubgroupSize = 64u;
constexpr uint32_t kGemvNumSubgroups = 4u;

GlobalWorkGrid pick_q4gsw_legacy_gemv_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, graph->sizes_of(out)));

  return GlobalWorkGrid({N / 2u, kGemvNumSubgroups, 1u}, kTiledWorkGrid);
}

LocalWorkGroup pick_q4gsw_legacy_gemv_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)gwg;
  (void)args;
  (void)resize_args;
  return LocalWorkGroup(kGemvSubgroupSize, kGemvNumSubgroups, 1u);
}

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

GlobalWorkGrid legacy_q4gsw_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);

  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));

  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));

  uint32_t N_per_tile = 8;
  uint32_t M_per_tile = 4;
  if (shader.kernel_name.find("coop") != std::string::npos) {
    M_per_tile = 1;
  }

  const uint32_t num_N_tiles = utils::div_up(N, N_per_tile);
  const uint32_t num_M_tiles = utils::div_up(M, M_per_tile);

  return GlobalWorkGrid({num_N_tiles, num_M_tiles, 1u}, kTiledWorkGrid);
}

LocalWorkGroup legacy_q4gsw_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return LocalWorkGroup(1u, 1u, 64u);
  }
  return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
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

ValueRef legacy_prepack_q4gsw_weight(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  const int64_t qmat2_width = qmat2_orig_sizes.at(ndim - 1);
  const int64_t qmat2_height = qmat2_orig_sizes.at(ndim - 2);

  const int64_t K = qmat2_width * 2;
  const int64_t N = qmat2_height;

  VK_CHECK_COND(K % 8 == 0);

  const int64_t N_per_block = 8;
  const int64_t K_per_block = 4;

  const int64_t num_blocks_K = utils::div_up(K, K_per_block);
  const int64_t num_blocks_N = utils::div_up(N, N_per_block);

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

  ValueRef cached = graph.get_cached_prepack(qmat2_data, kernel_name);
  if (is_valid(cached)) {
    return cached;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkapi::kInt, storage_type, utils::kWidthPacked);

  const GlobalWorkGrid gwg(
      {utils::safe_downcast<uint32_t>(utils::div_up(num_blocks_K, int64_t(2))),
       utils::safe_downcast<uint32_t>(num_blocks_N),
       1u},
      kTiledWorkGrid);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      gwg,
      graph.create_lwg(gwg),
      qmat2_data,
      qmat2,

      {},

      {},

      {graph.sizes_pc_of(qmat2),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  graph.cache_prepack(qmat2_data, kernel_name, qmat2);
  return qmat2;
}

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
      legacy_q4gsw_gwg,
      legacy_q4gsw_lwg,

      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},

      param_buffers,

      {},

      {apply_bias, K4_per_group},

      {kDummyValueRef, weight_data},
      legacy_q4gsw_resize_linear_node));
}

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
  using PickGwgFn = GlobalWorkGrid (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickLwgFn = LocalWorkGroup (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const GlobalWorkGrid&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);

  PickShaderFn pick_shader = nullptr;
  PickGwgFn pick_gwg = nullptr;
  PickLwgFn pick_lwg = nullptr;

  switch (kind) {
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER:
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G1W64:
      pick_shader =
          pick_forced_shader_coop_kc<TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER>;
      pick_gwg = pick_q4gsw_coop_gwg<1u, 64u>;
      pick_lwg = pick_q4gsw_coop_lwg<1u, 64u>;
      break;
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16:
      pick_shader = pick_forced_shader_coop_kc<
          TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G4W16>;
      pick_gwg = pick_q4gsw_coop_gwg<4u, 16u>;
      pick_lwg = pick_q4gsw_coop_lwg<4u, 16u>;
      break;
    case TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8:
      pick_shader = pick_forced_shader_coop_kc<
          TestKernelKind::GEMV_COOP_W_4X8_NC_BUFFER_G8W8>;
      pick_gwg = pick_q4gsw_coop_gwg<8u, 8u>;
      pick_lwg = pick_q4gsw_coop_lwg<8u, 8u>;
      break;
    default:
      VK_THROW("add_q4gsw_linear_coop_kc_forced_node: non-coop kind");
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader,
      pick_gwg,
      pick_lwg,
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
  using PickGwgFn = GlobalWorkGrid (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);
  using PickLwgFn = LocalWorkGroup (*)(
      ComputeGraph*,
      const vkapi::ShaderInfo&,
      const GlobalWorkGrid&,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&);

  PickShaderFn pick_shader = nullptr;
  PickGwgFn pick_gwg = nullptr;
  PickLwgFn pick_lwg = nullptr;

  switch (kind) {
    case TestKernelKind::GEMM_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMM_W_4X8>;
      pick_gwg = pick_q4gsw_linear_gemm_gwg;
      pick_lwg = pick_q4gsw_linear_gemm_lwg;
      break;
    case TestKernelKind::GEMV_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMV_W_4X8>;
      pick_gwg = pick_q4gsw_legacy_gemv_gwg;
      pick_lwg = pick_q4gsw_legacy_gemv_lwg;
      break;
    case TestKernelKind::GEMM_TIN_W_4X8:
      pick_shader = pick_forced_shader<TestKernelKind::GEMM_TIN_W_4X8>;
      pick_gwg = pick_q4gsw_linear_tin_gemm_gwg;
      pick_lwg = pick_q4gsw_linear_tin_gemm_lwg;
      break;
    case TestKernelKind::GEMV_W_4X8_NOSG:
      pick_shader = pick_forced_shader<TestKernelKind::GEMV_W_4X8_NOSG>;
      pick_gwg = pick_q4gsw_legacy_gemv_gwg;
      pick_lwg = pick_q4gsw_legacy_gemv_lwg;
      break;
    case TestKernelKind::PROD:
    default:
      VK_THROW("PROD kind must be dispatched via production entry points");
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader,
      pick_gwg,
      pick_lwg,
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

namespace shader_profiling {
namespace {
template <int32_t Selector, bool Gemv>
Json build_q4_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Q4 routes accept no options");
  if (args.size() != 6)
    throw std::invalid_argument("Q4 route requires six canonical arguments");
  const auto* adapter = graph.context()->adapter_ptr();
  if constexpr (Selector == 1 && Gemv) {
    if (!adapter->supports_required_subgroup_size_for_compute() ||
        adapter->min_subgroup_size() > 64 ||
        adapter->max_subgroup_size() < 64 ||
        !adapter->supports_subgroup_compute_basic() ||
        !adapter->supports_subgroup_compute_ballot() ||
        !adapter->supports_subgroup_compute_shuffle())
      throw UnsupportedRoute(
          "gemv_sg requires supported subgroup64 basic/ballot/shuffle");
  }
  if constexpr ((Selector == 1 || Selector == 2) && Gemv) {
    if (adapter->max_compute_workgroup_invocations() < 256 ||
        adapter->max_compute_workgroup_size()[0] < 64 ||
        adapter->max_compute_workgroup_size()[1] < 4)
      throw UnsupportedRoute("Legacy GEMV requires a 64x4 workgroup");
  }
  if constexpr (Gemv) {
    if (graph.sizes_of(args[0]).at(0) != 1)
      throw UnsupportedRoute("GEMV route requires M=1");
  }
  add_fpa_q4gsw_linear_node(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      Selector,
      Gemv,
      args[5]);
  const char* name = Selector == 3 ? "legacy"
      : Gemv                       ? (Selector == 1        ? "gemv_sg"
                                          : Selector == 2  ? "gemv_nosg"
                                          : Selector == 13 ? "gemv_coop"
                                          : Selector == 14 ? "gemv_g1w64"
                                          : Selector == 15 ? "gemv_g4w16"
                                                           : "gemv_g8w8")
                                   : (Selector == 1 ? "gemm" : "gemm_tin");
  return {{"name", name}, {"options", options}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemm, (build_q4_route<1, false>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemm_tin, (build_q4_route<2, false>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, legacy, (build_q4_route<3, false>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_sg, (build_q4_route<1, true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_nosg, (build_q4_route<2, true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_coop, (build_q4_route<13, true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_g1w64, (build_q4_route<14, true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_g4w16, (build_q4_route<15, true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.q4gsw_linear.default, gemv_g8w8, (build_q4_route<16, true>));
}
} // namespace
} // namespace shader_profiling
} // namespace vkcompute
