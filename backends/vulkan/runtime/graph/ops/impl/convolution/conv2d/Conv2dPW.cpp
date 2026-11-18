/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

namespace {

// Below this many wide tile invocations the GPU is underfilled, and Half
// convolutions are faster with the narrow tile. The threshold sits between the
// largest swept Half grid that was faster with the narrow tile on most GPUs
// (2560 wide tile invocations) and the smallest faster with the wide tile on
// every GPU (12288); grids in between were not swept.
constexpr uint64_t kMinWideTileInvocations = 4096;

// Row-aligned work groups on Mali: their thread count, and the pixel and
// channel tile counts of the convolutions that take them.
constexpr uint32_t kRowAlignedThreads = 896;
constexpr uint32_t kRowAlignedMinPixelTiles = 8192;
constexpr uint32_t kRowAlignedMinChannelTiles = 8;
constexpr uint32_t kRowAlignedMaxChannelTiles = 64;

// Pixel tile width of every variant; must match conv2d_pw_tiled.yaml.
constexpr uint32_t kTileW = 2;

/*
 * The 2x2-pixel x 8-channel tile issues half as many input and weight loads per
 * FMA as the 2x1-pixel x 4-channel tile. Float uses it on every GPU except
 * Mali, where the wide Float tile spills registers in the Mali offline compiler
 * and was 2-31% slower than the narrow tile.
 */
bool use_wide_pw_tile(ComputeGraph* graph, const ValueRef out) {
  if (graph->dtype_of(out) != vkapi::kHalf) {
    return !graph->device_is_mali();
  }
  const uint64_t M = uint64_t(graph->size_at<uint32_t>(-1, out)) *
      graph->size_at<uint32_t>(-2, out);
  const uint64_t N4 = utils::div_up_4(graph->size_at<uint32_t>(-3, out));
  return utils::div_up(M, uint64_t(4)) * utils::div_up(N4, uint64_t(2)) >=
      kMinWideTileInvocations;
}

/*
 * Half wide tile convolutions took 0.829, 0.949 and 0.949 of the time on Adreno
 * 840, 750 and 740 (0.731 to 1.032 per shape) with the weights in a texture and
 * accumulators that start from the bias and sum each output texel's products
 * in a temporary, over 1x1 convolutions of 32 to 512 channels with 189x252 to
 * 1008x756 outputs. Mali-G715 and Xclipse 940 took 1.80 and 1.28 times as long
 * with them.
 */
bool use_folded_accumulation(ComputeGraph* graph, const ValueRef out) {
  return graph->dtype_of(out) == vkapi::kHalf && graph->device_is_adreno() &&
      use_wide_pw_tile(graph, out);
}

/*
 * On Mali-G715, 1x1 convolutions with at least as many 4-channel output slices
 * as input slices, 8, 16, 32 or 64 channel tiles and at least 8192 pixel tiles
 * took 0.771 to 1.008 of the time of the other picks with work groups of up to
 * 896 threads whose y spans the channel tiles and whose x divides the row of
 * pixel tiles. Other shapes took up to 1.11 of the time with such work groups
 * and keep the other picks. Adreno 740 to 840 took 1.03 to 1.05 of the time
 * with the same rule (up to 1.33 per shape), and Xclipse 940 up to 1.99 per
 * shape.
 */
bool use_row_aligned_lwg(
    ComputeGraph* graph,
    const ValueRef in,
    const ValueRef out,
    const GlobalWorkGrid& gwg) {
  const uint32_t n_tiles = gwg[1];
  return graph->device_is_mali() && gwg[0] >= kRowAlignedMinPixelTiles &&
      n_tiles >= kRowAlignedMinChannelTiles &&
      n_tiles <= kRowAlignedMaxChannelTiles &&
      (n_tiles & (n_tiles - 1u)) == 0u &&
      utils::div_up_4(graph->size_at<uint32_t>(-3, out)) >=
      utils::div_up_4(graph->size_at<uint32_t>(-3, in));
}

// The largest x up to max_x that divides row_tiles, or max_x if that is below
// half of it.
uint32_t pick_row_divisor(const uint32_t row_tiles, const uint32_t max_x) {
  uint32_t x = max_x;
  while (row_tiles % x != 0u) {
    --x;
  }
  return x * 2u >= max_x ? x : max_x;
}

} // namespace

void resize_conv2d_pw_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  std::vector<int64_t> self_sizes = graph->sizes_of(self);
  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));
  const auto& weight_sizes = weight_ref->sizes;

  const auto stride_list = graph->get_int_list(extra_args.at(1));
  const auto padding_list = graph->get_int_list(extra_args.at(2));

  const int64_t stride_h = stride_list->at(0);
  const int64_t stride_w = stride_list->at(1);
  const int64_t padding_h = padding_list->at(0);
  const int64_t padding_w = padding_list->at(1);

  const int64_t in_h = self_sizes.at(self_sizes.size() - 2);
  const int64_t in_w = self_sizes.at(self_sizes.size() - 1);

  // For 1x1 kernel with dilation=1: out = (in + 2*padding - 1) / stride + 1
  const int64_t out_h = (in_h + 2 * padding_h - 1) / stride_h + 1;
  const int64_t out_w = (in_w + 2 * padding_w - 1) / stride_w + 1;

  std::vector<int64_t> new_out_sizes = self_sizes;
  new_out_sizes.at(self_sizes.size() - 3) = weight_sizes.at(0);
  new_out_sizes.at(self_sizes.size() - 2) = out_h;
  new_out_sizes.at(self_sizes.size() - 1) = out_w;

  graph->virtual_resize(out, new_out_sizes);
  check_conv2d_residual_on_resize(graph, out, extra_args);
}

vkapi::ShaderInfo pick_conv2d_pw_tiled_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef packed_weight = args.at(1).refs.at(1);

  std::string kernel_name = "conv2d_pw_tiled";
  kernel_name.reserve(kShaderNameReserve);
  if (use_wide_pw_tile(graph, out)) {
    kernel_name += "_4x8";
  }
  if (graph->is_buffer_storage(packed_weight)) {
    kernel_name += "_bufw";
  }
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

GlobalWorkGrid pick_conv2d_pw_tiled_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  uint32_t W = graph->size_at<uint32_t>(-1, out);
  uint32_t H = graph->size_at<uint32_t>(-2, out);
  uint32_t C_out = graph->size_at<uint32_t>(-3, out);
  uint32_t N4 = utils::div_up_4(C_out);
  // Tile sizes must match conv2d_pw_tiled.yaml.
  const bool wide = use_wide_pw_tile(graph, out);
  const uint32_t tile_n4 = wide ? 2u : 1u;
  const uint32_t tile_h = wide ? 2u : 1u;
  const uint32_t n_tiles = utils::div_up(N4, tile_n4);
  const uint32_t m_tiles = utils::div_up(W, kTileW) * utils::div_up(H, tile_h);
  // The wide tile variant puts pixel tiles on x.
  if (wide) {
    return GlobalWorkGrid({m_tiles, n_tiles, 1u}, kTiledWorkGrid);
  }
  return GlobalWorkGrid({n_tiles, m_tiles, 1u}, kTiledWorkGrid);
}

/*
 * Wide tile work groups span the channel tiles of a row of pixel tiles, so the
 * input they share is read through one core's cache: y covers the channel tiles
 * (rounded up to a power of two) while x keeps at least min_x pixel tiles, and
 * x takes the remaining threads. Row-aligned work groups instead take more
 * threads, cover every channel tile in y, and pick an x that divides the row of
 * pixel tiles, which can leave fewer than min_x pixel tiles. Pixel tiles are
 * dispatched on x alone, so x also grows where the thread count allows to keep
 * the work group count on x within the device limit. Row-aligned work groups
 * are only taken where their x can do so.
 *
 * The thread count and min_x per GPU family come from sweeping 12 local sizes
 * (32 to 512 threads with min_x 4 to 32, and an 8x8 square) over 57 Float and
 * Half 1x1 convolutions with 7x7 to 1008x756 outputs on Mali, Adreno and
 * Xclipse phones. On the large convolutions, Mali was 5-8% faster with 512
 * threads than with 128, Adreno was 1-10% slower with 512 than with 128, and
 * Xclipse was 30% slower with 128 than with 64.
 */
LocalWorkGroup pick_conv2d_pw_tiled_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  if (use_wide_pw_tile(graph, out)) {
    const vkapi::Adapter* const adapter = graph->context()->adapter_ptr();
    const auto fit_threads = [adapter](uint32_t nthreads) {
      while (nthreads > adapter->max_compute_workgroup_invocations() ||
             nthreads > adapter->max_compute_workgroup_size()[0]) {
        nthreads /= 2u;
      }
      return nthreads;
    };
    const uint32_t min_count_x =
        utils::div_up(gwg[0], adapter->max_compute_workgroup_count()[0]);
    if (use_row_aligned_lwg(graph, args.at(1).refs.at(0), out, gwg)) {
      const uint32_t max_x = fit_threads(kRowAlignedThreads) / gwg[1];
      if (max_x >= min_count_x) {
        const uint32_t row_tiles =
            utils::div_up(graph->size_at<uint32_t>(-1, out), kTileW);
        const uint32_t x = pick_row_divisor(row_tiles, max_x);
        return LocalWorkGroup(x >= min_count_x ? x : max_x, gwg[1], 1u);
      }
    }
    const bool mali = graph->device_is_mali();
    const bool adreno = graph->device_is_adreno();
    const uint32_t nthreads = fit_threads(mali ? 512u : adreno ? 128u : 64u);
    uint32_t min_x = mali || adreno ? 16u : 4u;
    while (min_x < min_count_x && min_x < nthreads) {
      min_x *= 2u;
    }
    // y stays at or below nthreads / min_x (32 at most), within the 128 that
    // Vulkan guarantees for the second work group dimension.
    uint32_t y = 1u;
    while (y < gwg[1] && y * min_x < nthreads) {
      y *= 2u;
    }
    return LocalWorkGroup(nthreads / y, y, 1u);
  }
  // On Mali, a work group of pixel tiles that share one channel tile lets each
  // warp read a single weight stream instead of one per channel tile.
  if (graph->device_is_mali() && gwg[1] >= 64u) {
    return LocalWorkGroup(1u, 64u, 1u);
  }
  return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
}

//
// Prepack nodes
//

struct PackParams {
  int32_t N;
  int32_t K;
  int32_t B;
  int32_t is_transposed;
};

ValueRef prepack_conv2d_pw_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const ValueRef out) {
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t N = weight_sizes.at(0); // C_out
  const int64_t K = weight_sizes.at(1); // C_in
  const int64_t N4 = utils::div_up(N, int64_t(4));
  const int64_t K4 = utils::div_up(K, int64_t(4));

  const int64_t output_height = K4;
  const int64_t output_width = N4 * 4 * 4;

  // Buffer weights are faster than texture weights on every GPU measured. The
  // folded accumulation reads texture weights, the only storage it was measured
  // with. Half buffers need 16-bit storage support.
  utils::StorageType weight_storage = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  const bool buffer_weights = !use_folded_accumulation(&graph, out) &&
      (graph.dtype_of(weight_data) != vkapi::kHalf ||
       graph.float16_buffers_enabled());
  if (buffer_weights || output_width / 4 > max_extent ||
      static_cast<uint32_t>(output_height) > max_extent) {
    weight_storage = utils::kBuffer;
  }

  ValueRef packed_weight = graph.add_tensor(
      {output_height, output_width},
      graph.dtype_of(weight_data),
      weight_storage,
      utils::kWidthPacked);

  const GlobalWorkGrid gwg(
      {utils::safe_downcast<uint32_t>(N4),
       utils::safe_downcast<uint32_t>(K4),
       1u},
      kTiledWorkGrid);

  PackParams pack_params{
      utils::safe_downcast<int32_t>(N), utils::safe_downcast<int32_t>(K), 1, 1};

  std::string pack_kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(pack_kernel_name, weight_storage);
  add_dtype_suffix(pack_kernel_name, graph.dtype_of(packed_weight));
  add_dtype_suffix(pack_kernel_name, graph.get_staging_dtype_for(weight_data));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(pack_kernel_name),
      gwg,
      graph.create_lwg(gwg),
      weight_data,
      packed_weight,
      {},
      {},
      {PushConstantDataInfo(&pack_params, sizeof(PackParams))}));

  return packed_weight;
}

//
// Dispatch nodes
//

void add_conv2d_pw_tiled_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const ValueRef weight_data,
    const Conv2dEpilogue& epilogue,
    const Conv2dEpilogueArgs& epilogue_args) {
  int32_t stride_h, stride_w, padding_h, padding_w;
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    stride_h = utils::safe_downcast<int32_t>(stride_list->at(0));
    stride_w = utils::safe_downcast<int32_t>(stride_list->at(1));
    padding_h = utils::safe_downcast<int32_t>(padding_list->at(0));
    padding_w = utils::safe_downcast<int32_t>(padding_list->at(1));
  }

  bool s1p0 =
      stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0;

  utils::ivec4 stride_padding{stride_h, stride_w, padding_h, padding_w};

  const Conv2dEpilogueParams epilogue_params = create_conv2d_epilogue_params(
      epilogue, graph.dtype_of(out) == vkapi::kHalf);

  vkapi::SpecVarList spec_vars = {s1p0 ? 1 : 0};
  append_conv2d_epilogue_spec_vars(spec_vars, epilogue);
  // The weight storage and folded_accumulation are fixed here, but the tile is
  // picked again on resize. Only the wide tile texture weight Half variant
  // declares folded_accumulation; the other variants ignore it.
  spec_vars.append(vkapi::SpecVar(use_folded_accumulation(&graph, out)));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_pw_tiled_shader,
      pick_conv2d_pw_tiled_gwg,
      pick_conv2d_pw_tiled_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{in,
         packed_weight,
         packed_bias,
         epilogue_args.residual,
         epilogue_args.scale},
        vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      // Push Constants
      {PushConstantDataInfo(&stride_padding, sizeof(stride_padding)),
       PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params))},
      // Specialization Constants
      spec_vars,
      // Resize Args
      {weight_data, stride, padding, epilogue.residual},
      // Resizing Logic
      resize_conv2d_pw_tiled_node));
}

//
// High level operator impl
//

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const Conv2dEpilogue& epilogue) {
  ValueRef packed_weight = prepack_conv2d_pw_weight(graph, weight_data, out);

  ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      transposed_val,
      utils::kTexture2D,
      utils::kWidthPacked);
  const ValueRef packed_scale = prepack_conv2d_residual_scale(
      graph, epilogue, weight_data, transposed_val);

  check_conv_args(graph, in, out);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  if (in_sizes.at(0) > 1) {
    VK_THROW("conv2d: input batch size > 1 is not supported yet!");
  }

  add_conv2d_pw_tiled_node(
      graph,
      in,
      packed_weight,
      packed_bias,
      stride,
      padding,
      out,
      weight_data,
      epilogue,
      Conv2dEpilogueArgs(epilogue, in, packed_bias, packed_scale));
}

} // namespace vkcompute
