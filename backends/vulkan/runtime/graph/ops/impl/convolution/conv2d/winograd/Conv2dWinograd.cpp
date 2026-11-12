/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/winograd/Conv2dWinograd.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>

namespace vkcompute {

namespace {

// Every configuration with shaders
constexpr Conv2dWinogradConfig kConv2dWinogradConfigs[] = {
    kConv2dWinogradF4x4_3x3};

// Shader name suffix of `config`, f<m>x<m>_<r>x<r>
std::string config_suffix(const Conv2dWinogradConfig& config) {
  const std::string m = std::to_string(config.out_tile);
  const std::string r = std::to_string(config.kernel_size);
  return "_f" + m + "x" + m + "_" + r + "x" + r;
}

// Bound on the bytes of each of the two scratch tensors. Larger outputs are
// computed in chunks of tile rows. At 128 channels and 189x252, one 28 MB chunk
// took 3% less time than two on Adreno, and two took less than one on Mali.
int64_t scratch_budget_bytes(ComputeGraph& graph) {
  return (graph.device_is_adreno() ? 32 : 16) * 1024 * 1024;
}

// The scratch tensors V and M are textures with texel (t, e * C / 4 + c4) for
// tile t of a chunk, element e and channel slice c4, where C is the input and
// the output channels respectively. Returns the height of the taller one.
int64_t scratch_height(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef weight_data) {
  return config.tile_elements() *
      utils::div_up_4(std::max(
          graph.size_at<int64_t>(0, weight_data),
          graph.size_at<int64_t>(1, weight_data)));
}

// The most tile rows of a chunk within the texture limits, 0 if none fit
int64_t max_chunk_rows(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data) {
  const int64_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  if (scratch_height(graph, config, weight_data) > max_extent) {
    return 0;
  }
  return max_extent / config.tiles_of(graph.size_at<int64_t>(-1, in));
}

// Tile rows per chunk, so that each scratch tensor fits the budget and the
// texture limits.
int64_t chunk_rows_of(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data) {
  const int64_t row_bytes = scratch_height(graph, config, weight_data) * 4 *
      config.tiles_of(graph.size_at<int64_t>(-1, in)) *
      int64_t(vkapi::element_size(graph.dtype_of(in)));
  return std::clamp(
      std::min(
          scratch_budget_bytes(graph) / row_bytes,
          max_chunk_rows(graph, config, in, weight_data)),
      int64_t(1),
      config.tiles_of(graph.size_at<int64_t>(-2, in)));
}

// Adreno reads the weights faster from a texture.
bool use_buffer_weights(ComputeGraph& graph) {
  return !graph.device_is_adreno() && graph.float16_buffers_enabled();
}

struct ChunkTiles {
  uint32_t x;
  uint32_t rows;
};

// The resize args of every Winograd node start with the input, the chunk's
// first tile row, the tile rows of a chunk, and the output tile side and kernel
// size of the configuration.
Conv2dWinogradConfig config_of(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  return {graph->get_int(resize_args.at(3)), graph->get_int(resize_args.at(4))};
}

// The tile columns and the live tile rows of a chunk. Chunks past the last tile
// row of a resized input have none.
ChunkTiles chunk_tiles(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  const Conv2dWinogradConfig config = config_of(graph, resize_args);
  const ValueRef in = resize_args.at(0);
  const int64_t first_row = graph->get_int(resize_args.at(1));
  const int64_t chunk_rows = graph->get_int(resize_args.at(2));
  const int64_t rows = std::clamp(
      config.tiles_of(graph->size_at<int64_t>(-2, in)) - first_row,
      int64_t(0),
      chunk_rows);
  return {
      utils::safe_downcast<uint32_t>(
          config.tiles_of(graph->size_at<int64_t>(-1, in))),
      utils::safe_downcast<uint32_t>(rows)};
}

// Work groups of the transforms span tile columns (x) and channel slices (z),
// but not y, so that the rows a work group transforms are uniform. On Adreno
// 750, `invocations` filled with tile columns first ran fastest. On Mali-G715,
// one channel slice with an invocation per tile column did; on narrow planes it
// took up to 2.9x less time than 128 invocations along x.
LocalWorkGroup fit_conv2d_winograd_transform_lwg(
    ComputeGraph* graph,
    const GlobalWorkGrid& gwg,
    const uint32_t invocations) {
  if (graph->device_is_mali()) {
    uint32_t x = 1u;
    while (x < gwg[0] && x < 128u) {
      x *= 2u;
    }
    return LocalWorkGroup(x, 1u, 1u);
  }
  // The weights give x every exponent of `invocations` before z gets one.
  LocalWorkGroup lwg(LwgShape(7u, 0u, 1u), invocations);
  lwg.fit_to_global(gwg);
  return lwg;
}

//
// Input transform
//

vkapi::ShaderInfo pick_conv2d_winograd_input_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  std::string kernel_name = "conv2d_winograd_input";
  kernel_name.reserve(kShaderNameReserve);
  if (graph->samples_zero_outside_tensor(args.at(1).refs.at(0))) {
    kernel_name += "_border";
  }
  kernel_name += config_suffix(config_of(graph, resize_args));
  add_dtype_suffix(kernel_name, graph->dtype_of(args.at(0).refs.at(0)));
  return VK_KERNEL_FROM_STR(kernel_name);
}

GlobalWorkGrid pick_conv2d_winograd_input_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ChunkTiles tiles = chunk_tiles(graph, resize_args);
  if (tiles.rows == 0u) {
    return GlobalWorkGrid({0u, 0u, 0u}, kTiledWorkGrid);
  }
  // An invocation per group of rows of V of each tile: the row of point 0, the
  // two rows of each pair +-p, and the row of infinity
  const uint32_t row_groups = utils::safe_downcast<uint32_t>(
      config_of(graph, resize_args).in_tile() / 2 + 1);
  return GlobalWorkGrid(
      {tiles.x,
       tiles.rows * row_groups,
       utils::div_up_4(graph->size_at<uint32_t>(-3, args.at(1).refs.at(0)))},
      kTiledWorkGrid);
}

LocalWorkGroup pick_conv2d_winograd_input_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  // On Mali-G715, 128 invocations along x took 11% less time than fitting them
  // to planes 63 tiles wide.
  if (graph->device_is_mali() && gwg[0] > 32u) {
    return LocalWorkGroup(128u, 1u, 1u);
  }
  return fit_conv2d_winograd_transform_lwg(graph, gwg, 64u);
}

void add_conv2d_winograd_input_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef v,
    const ValueRef first_row,
    const ValueRef chunk_rows,
    const ValueRef out_tile,
    const ValueRef kernel_size) {
  const utils::ivec2 chunk{
      utils::safe_downcast<int32_t>(graph.get_int(first_row)),
      utils::safe_downcast<int32_t>(graph.get_int(chunk_rows))};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_winograd_input_shader,
      pick_conv2d_winograd_input_gwg,
      pick_conv2d_winograd_input_lwg,
      {{v, vkapi::kWrite}, {in, vkapi::kRead}},
      {graph.sizes_ubo(in)},
      {PushConstantDataInfo(&chunk, sizeof(chunk))},
      {},
      {in, first_row, chunk_rows, out_tile, kernel_size}));
}

//
// Batched GEMM
//

ValueRef prepack_conv2d_winograd_weight(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef weight_data,
    const utils::StorageType storage) {
  const int64_t N = graph.size_at<int64_t>(0, weight_data);
  const int64_t K = graph.size_at<int64_t>(1, weight_data);
  const utils::ivec2 dims{
      utils::safe_downcast<int32_t>(N), utils::safe_downcast<int32_t>(K)};
  // A batch per element of a transformed tile
  return prepack_fp_blocked_weight(
      graph,
      weight_data,
      "pack_conv2d_winograd_weight" + config_suffix(config),
      N,
      K,
      config.tile_elements(),
      storage,
      PushConstantDataInfo(&dims, sizeof(dims)));
}

vkapi::ShaderInfo pick_conv2d_winograd_gemm_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  std::string kernel_name = "conv2d_winograd_gemm";
  kernel_name.reserve(kShaderNameReserve);
  if (graph->is_buffer_storage(args.at(1).refs.at(1))) {
    kernel_name += "_bufw";
  }
  add_dtype_suffix(kernel_name, graph->dtype_of(args.at(0).refs.at(0)));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Each invocation computes 4 tiles for 8 output channels of one element; must
// match conv2d_winograd_gemm.glsl.
GlobalWorkGrid pick_conv2d_winograd_gemm_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const ChunkTiles tiles = chunk_tiles(graph, resize_args);
  if (tiles.rows == 0u) {
    return GlobalWorkGrid({0u, 0u, 0u}, kTiledWorkGrid);
  }
  const uint32_t N4 =
      utils::div_up_4(graph->size_at<uint32_t>(0, resize_args.at(5)));
  return GlobalWorkGrid(
      {utils::div_up(tiles.x * tiles.rows, 4u),
       utils::div_up(N4, 2u),
       utils::safe_downcast<uint32_t>(
           config_of(graph, resize_args).tile_elements())},
      kTiledWorkGrid);
}

LocalWorkGroup pick_conv2d_winograd_gemm_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  // 16 tile groups by 16 column groups ran fastest on Adreno, and 4 by 16 on
  // Mali.
  const bool adreno = graph->device_is_adreno();
  const uint32_t nthreads = adreno ? 256u : 64u;
  const uint32_t min_x = adreno ? 16u : 4u;
  uint32_t y = 1u;
  while (y < gwg[1] && y * min_x < nthreads) {
    y *= 2u;
  }
  return LocalWorkGroup(nthreads / y, y, 1u, nthreads);
}

void add_conv2d_winograd_gemm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef v,
    const ValueRef packed_weight,
    const ValueRef m,
    const ValueRef weight_data,
    const ValueRef first_row,
    const ValueRef chunk_rows,
    const ValueRef out_tile,
    const ValueRef kernel_size) {
  const vkapi::SpecVarList spec_vars = {utils::safe_downcast<int32_t>(
      utils::div_up_4(graph.size_at<int64_t>(0, weight_data)))};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_winograd_gemm_shader,
      pick_conv2d_winograd_gemm_gwg,
      pick_conv2d_winograd_gemm_lwg,
      {{m, vkapi::kWrite}, {{v, packed_weight}, vkapi::kRead}},
      {graph.sizes_ubo(in)},
      {},
      spec_vars,
      {in, first_row, chunk_rows, out_tile, kernel_size, weight_data}));
}

//
// Output transform
//

GlobalWorkGrid pick_conv2d_winograd_output_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ChunkTiles tiles = chunk_tiles(graph, resize_args);
  if (tiles.rows == 0u) {
    return GlobalWorkGrid({0u, 0u, 0u}, kTiledWorkGrid);
  }
  // An invocation per output row of each tile
  return GlobalWorkGrid(
      {tiles.x,
       tiles.rows *
           utils::safe_downcast<uint32_t>(
               config_of(graph, resize_args).out_tile),
       utils::div_up_4(graph->size_at<uint32_t>(-3, args.at(0).refs.at(0)))},
      kTiledWorkGrid);
}

LocalWorkGroup pick_conv2d_winograd_output_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  return fit_conv2d_winograd_transform_lwg(graph, gwg, 128u);
}

void resize_conv2d_winograd_output_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  std::vector<int64_t> out_sizes = graph->sizes_of(resize_args.at(0));
  out_sizes.at(1) = graph->size_at<int64_t>(0, resize_args.at(5));
  graph->virtual_resize(out, out_sizes);
  check_conv2d_residual_on_resize(graph, out, resize_args);
}

void add_conv2d_winograd_output_node(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef m,
    const ValueRef packed_bias,
    const ValueRef out,
    const ValueRef weight_data,
    const ValueRef first_row,
    const ValueRef chunk_rows,
    const ValueRef out_tile,
    const ValueRef kernel_size,
    const Conv2dEpilogue& epilogue,
    const Conv2dEpilogueArgs& epilogue_args) {
  const utils::ivec4 chunk{
      utils::safe_downcast<int32_t>(graph.get_int(first_row)),
      utils::safe_downcast<int32_t>(graph.get_int(chunk_rows)),
      0,
      0};
  const Conv2dEpilogueParams epilogue_params =
      create_conv2d_epilogue_params(epilogue, /*f16_epilogue=*/true);
  vkapi::SpecVarList spec_vars;
  append_conv2d_epilogue_spec_vars(spec_vars, epilogue);

  std::string kernel_name = "conv2d_winograd_output";
  kernel_name += config_suffix(config);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_conv2d_winograd_output_gwg,
      pick_conv2d_winograd_output_lwg,
      {{out, vkapi::kWrite},
       {{m, packed_bias, epilogue_args.residual, epilogue_args.scale},
        vkapi::kRead}},
      {graph.sizes_ubo(out)},
      {PushConstantDataInfo(&chunk, sizeof(chunk)),
       PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params))},
      spec_vars,
      {in,
       first_row,
       chunk_rows,
       out_tile,
       kernel_size,
       weight_data,
       epilogue.residual},
      resize_conv2d_winograd_output_node));
}

} // namespace

bool can_use_conv2d_winograd(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups) {
  if (transposed || groups != 1 || graph.dim_of(in) != 4 ||
      graph.size_at<int64_t>(0, in) != 1) {
    return false;
  }
  const Kernel2dParams& p = kernel_params;
  const int64_t k = config.kernel_size;
  const int64_t pad = (k - 1) / 2;
  // Only an odd kernel size keeps the output at the input's size.
  if (k % 2 == 0 || p.kernel_size[0] != k || p.kernel_size[1] != k ||
      p.stride[0] != 1 || p.stride[1] != 1 || p.dilation[0] != 1 ||
      p.dilation[1] != 1 || p.padding[0] != pad || p.padding[1] != pad) {
    return false;
  }
  if (graph.dtype_of(in) != vkapi::kHalf ||
      graph.dtype_of(out) != vkapi::kHalf ||
      graph.dtype_of(weight_data) != vkapi::kHalf ||
      !graph.is_standard_channels_packed_texture_tensor(in) ||
      !graph.is_standard_channels_packed_texture_tensor(out) ||
      graph.storage_type_of(in) != utils::kTexture3D ||
      graph.storage_type_of(out) != utils::kTexture3D) {
    return false;
  }
  return max_chunk_rows(graph, config, in, weight_data) > 0;
}

std::optional<Conv2dWinogradConfig> find_conv2d_winograd_config(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups) {
  for (const Conv2dWinogradConfig& config : kConv2dWinogradConfigs) {
    if (can_use_conv2d_winograd(
            graph,
            config,
            in,
            weight_data,
            out,
            kernel_params,
            transposed,
            groups)) {
      return config;
    }
  }
  return std::nullopt;
}

std::optional<Conv2dWinogradConfig> pick_conv2d_winograd_config(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups) {
  if (!(graph.device_is_adreno() || graph.device_is_mali())) {
    return std::nullopt;
  }
  const int64_t channels = std::min(
      graph.size_at<int64_t>(0, weight_data),
      graph.size_at<int64_t>(1, weight_data));
  const int64_t plane =
      graph.size_at<int64_t>(-1, in) * graph.size_at<int64_t>(-2, in);
  // Against the direct and im2col routes on Adreno 740, 750 and 830 and
  // Mali-G715, F(4x4, 3x3) took 0.2 to 0.9 times as long with 128 or more
  // channels, on planes of 32x32 and larger. It also won with 48 to 96
  // channels on large planes, but its f16 rounding error there cost DSR's
  // full-resolution 64-channel convs 2 to 3.5 dB of output PSNR. It was slower
  // on most shapes on Xclipse 940.
  if (channels >= 128 && plane >= 32 * 32 &&
      can_use_conv2d_winograd(
          graph,
          kConv2dWinogradF4x4_3x3,
          in,
          weight_data,
          out,
          kernel_params,
          transposed,
          groups)) {
    return kConv2dWinogradF4x4_3x3;
  }
  return std::nullopt;
}

void conv2d_winograd_impl(
    ComputeGraph& graph,
    const Conv2dWinogradConfig& config,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef out,
    const Conv2dEpilogue& epilogue,
    const std::optional<int64_t> chunk_rows_override,
    const std::optional<utils::StorageType> weight_storage_override) {
  const int64_t C_in =
      utils::align_up_4(graph.size_at<int64_t>(1, weight_data));
  const int64_t C_out =
      utils::align_up_4(graph.size_at<int64_t>(0, weight_data));
  const int64_t tiles_x = config.tiles_of(graph.size_at<int64_t>(-1, in));
  const int64_t tiles_y = config.tiles_of(graph.size_at<int64_t>(-2, in));
  const int64_t chunk_rows = std::min(
      chunk_rows_override.value_or(
          chunk_rows_of(graph, config, in, weight_data)),
      tiles_y);
  VK_CHECK_COND(
      chunk_rows > 0 &&
      chunk_rows <= max_chunk_rows(graph, config, in, weight_data));

  const ValueRef packed_weight = prepack_conv2d_winograd_weight(
      graph,
      config,
      weight_data,
      weight_storage_override.value_or(
          use_buffer_weights(graph) ? utils::kBuffer : utils::kTexture2D));
  const ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      /*transposed=*/false,
      utils::kTexture2D,
      utils::kWidthPacked);
  const ValueRef packed_scale = prepack_conv2d_residual_scale(
      graph, epilogue, weight_data, /*transposed=*/false);
  check_conv_args(graph, in, out);

  // Element e of slice c4 of tile t is texel (t, e * C / 4 + c4). Each texel
  // holds 4 channels, so the width-packed sizes are 4 times the tile count.
  TmpTensor v(
      &graph,
      {config.tile_elements() * C_in / 4, 4 * chunk_rows * tiles_x},
      graph.dtype_of(in),
      utils::kTexture2D,
      utils::kWidthPacked);
  TmpTensor m(
      &graph,
      {config.tile_elements() * C_out / 4, 4 * chunk_rows * tiles_x},
      graph.dtype_of(out),
      utils::kTexture2D,
      utils::kWidthPacked);

  const Conv2dEpilogueArgs epilogue_args(
      epilogue, in, packed_bias, packed_scale);
  const ValueRef chunk_rows_ref = graph.add_scalar<int64_t>(chunk_rows);
  const ValueRef out_tile = graph.add_scalar<int64_t>(config.out_tile);
  const ValueRef kernel_size = graph.add_scalar<int64_t>(config.kernel_size);
  for (int64_t row = 0; row < tiles_y; row += chunk_rows) {
    const ValueRef first_row = graph.add_scalar<int64_t>(row);
    add_conv2d_winograd_input_node(
        graph, in, v, first_row, chunk_rows_ref, out_tile, kernel_size);
    add_conv2d_winograd_gemm_node(
        graph,
        in,
        v,
        packed_weight,
        m,
        weight_data,
        first_row,
        chunk_rows_ref,
        out_tile,
        kernel_size);
    add_conv2d_winograd_output_node(
        graph,
        config,
        in,
        m,
        packed_bias,
        out,
        weight_data,
        first_row,
        chunk_rows_ref,
        out_tile,
        kernel_size,
        epilogue,
        epilogue_args);
  }
}

} // namespace vkcompute
