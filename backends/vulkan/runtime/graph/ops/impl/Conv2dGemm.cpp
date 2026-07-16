/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dGemm.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dIm2Col.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <optional>

namespace vkcompute {

namespace {

// Byte budget for the im2col scratch tensor. The full im2col matrix is
// [M, K_total] = M * K_total * elem bytes (M = H_out * W_out); at high
// resolution this reaches hundreds of MB (e.g. 144 MB FP32 for a
// [1,64,256,256] 3x3 conv), a non-reclaimable device-local allocation resident
// for the whole model lifetime — an OOM risk on memory-constrained mobile GPUs.
// Materializing the im2col in tiles of output-height rows caps the scratch to
// this budget regardless of resolution while preserving GEMM throughput (the
// GEMM inner loop is unchanged; only the live row count is bounded). Tunable:
// a larger budget means fewer tiles / dispatches per conv but more peak memory.
//
// This is a LOGICAL-size budget (oh_tile is derived from W_out * K_total *
// elem bytes). The physical texture2d / texture3d allocation rounds the packed
// dim up to whole texels (vec4) and adds image row / layer alignment, so actual
// device memory for the scratch can modestly exceed this figure. Treat it as a
// soft tuning knob, not a hard allocation ceiling.
constexpr int64_t kIm2colScratchBudgetBytes = 16 * 1024 * 1024;

//
// Weight handling
//

// Prepack the ORIGINAL serialized conv2d weight [C_out, C_in, K_h, K_w]
// directly on the GPU into the 4OC x 4IC blocked layout that conv2d_gemm.glsl
// loads via load_packed_weight_tile_with_checks. The serialized weight data is
// read as-is (never CPU-repacked); pack_conv2d_gemm_weight.glsl performs the
// im2col K-axis reorder (k = (ki * K_w + kj) * Cin_padded + ci, ci-padding
// lanes zeroed) and the 4x4 transpose in one pass.
//
// The packed output is byte-identical to the layout the generic
// prepack_fp_linear_weight (is_transposed=1) produced over a CPU-flattened
// [C_out, K_total] weight, so conv2d_gemm.glsl is unchanged.
ValueRef prepack_conv2d_gemm_weight(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  VK_CHECK_COND(w_sizes.size() == 4);
  const int64_t C_out = w_sizes[0];
  const int64_t C_in = w_sizes[1];
  const int64_t K_h = w_sizes[2];
  const int64_t K_w = w_sizes[3];

  const int64_t Cin_padded = utils::align_up_4(C_in);
  const int64_t K_total = K_h * K_w * Cin_padded;

  const int64_t N = C_out;
  const int64_t K = K_total;
  const int64_t N4 = utils::div_up(N, int64_t(4));
  const int64_t K4 = utils::div_up(K, int64_t(4));

  // Packed tensor: K4 rows, N4*4 vec4 elements per row (4OC x 4IC blocks).
  // kWidthPacked packs 4 scalars per texel, so width = N4*4*4 scalars.
  const int64_t output_height = K4;
  const int64_t output_width = N4 * 4 * 4;

  // The GEMM shader (conv2d_gemm.glsl) only reads the packed weight as a
  // texture2d. A buffer-backed packed weight would require a WEIGHT_BUFFER
  // codegen variant of conv2d_gemm.glsl (and its picker), which does not exist
  // yet.
  // TODO: if this check ever triggers for a real model, add buffer-backed
  // packed-weight support — a WEIGHT_BUFFER variant of conv2d_gemm.{glsl,yaml}
  // with the picker routed accordingly, plus the buffer variants restored in
  // pack_conv2d_gemm_weight.yaml.
  const utils::StorageType weight_storage = utils::kTexture2D;
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  VK_CHECK_COND(
      output_width / 4 <= max_extent &&
      utils::safe_downcast<uint32_t>(output_height) <= max_extent);

  ValueRef packed_weight = graph.add_tensor(
      {output_height, output_width},
      graph.dtype_of(weight_data),
      weight_storage,
      utils::kWidthPacked);

  const utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(N4),
      utils::safe_downcast<uint32_t>(K4),
      1u};

  // Push constants must be uploaded in <= 16-byte (one ivec4) chunks; the
  // shader's Block reads them back as dims0 / dims1. Layout must match
  // pack_conv2d_gemm_weight.glsl.
  const utils::ivec4 dims0{
      utils::safe_downcast<int32_t>(N),
      utils::safe_downcast<int32_t>(K),
      utils::safe_downcast<int32_t>(C_in),
      utils::safe_downcast<int32_t>(Cin_padded)};
  const utils::ivec4 dims1{
      utils::safe_downcast<int32_t>(K_h),
      utils::safe_downcast<int32_t>(K_w),
      0,
      0};

  std::string kernel_name = "pack_conv2d_gemm_weight";
  add_storage_type_suffix(kernel_name, weight_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(weight_data));
  add_dtype_suffix(kernel_name, graph.get_staging_dtype_for(weight_data));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      weight_data,
      packed_weight,
      {},
      {},
      {PushConstantDataInfo(&dims0, sizeof(dims0)),
       PushConstantDataInfo(&dims1, sizeof(dims1))}));

  return packed_weight;
}

//
// GEMM dispatch
//

vkapi::ShaderInfo pick_conv2d_gemm_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  // The im2col tensor's storage selects the input-load codegen variant of
  // conv2d_gemm: texture2d vs buffer.
  const ValueRef im2col_in = args.at(1).refs.at(0);

  std::string kernel_name = "conv2d_gemm";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(im2col_in));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

// resize_args = { in, weight_data, stride, padding, dilation, oh_tile,
//                 oh_offset }
// resize_args[5] / [6] carry the raw oh_tile / oh_offset VALUES (not ValueRef
// handles): both are build-time constants, so packing the ints directly into
// the slots avoids materializing graph Values for them. Read with static_cast,
// never get_int.
utils::uvec3 pick_conv2d_gemm_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args.at(0).refs.at(0);
  const uint32_t W = graph->size_at<uint32_t>(-1, out);
  const uint32_t H_out = graph->size_at<uint32_t>(-2, out);
  const uint32_t C_out = graph->size_at<uint32_t>(-3, out);
  const int32_t oh_tile = static_cast<int32_t>(resize_args.at(5));
  const int32_t oh_offset = static_cast<int32_t>(resize_args.at(6));
  // Dead-tile skip: when a runtime down-resize shrinks H_out so this tile's
  // first output row (oh_offset) is already past the live H_out, the whole tile
  // is dead. Return a zero-sized global wg so DispatchNode::encode() emits zero
  // workgroups for it (it explicitly skips any dispatch whose global wg has a 0
  // component) — no GEMM work, no im2col read. trigger_resize() recomputes this
  // and re-encodes the command buffer when the dispatch grid changes, so the
  // skip tracks the dynamic shape. (The num_tiles count is still fixed at build
  // time; this only zeroes the work of tiles that fall off the live region.)
  if (oh_offset >= static_cast<int32_t>(H_out)) {
    return {0u, 0u, 0u};
  }
  // Every live tile dispatches oh_tile output-height rows (oh_tile * W per-tile
  // M); trailing threads past the real H_out no-op in the shader.
  const uint32_t M_tile = static_cast<uint32_t>(oh_tile) * W;
  const uint32_t N4 = utils::div_up_4(C_out);
  // TILE_N4=1, TILE_M=4
  return {N4, utils::div_up(M_tile, 4u), 1};
}

// Recompute the conv output sizes from the current input shape and resize the
// output tensor. This is the load-bearing resize for the im2col/GEMM path:
// under dynamic shapes the graph is built for the upper-bound input, so on
// trigger_resize() the output must be recomputed from the real input or it
// stays frozen at the upper bound (producing garbage downstream). Every tile's
// GEMM node shares this resize (each writes a different oh-row window of the
// same full output tensor).
//
// The GEMM shader derives W_out / H_out and the spatial store coordinates from
// the (now-refreshed) out_sizes UBO, so resizing `out` here is sufficient to
// make every tile track the dynamic shape — no push-constant update is needed
// (oh_offset / oh_tile are shape-independent). The global workgroup picker
// reads oh_tile (resize_args[5]) and oh_offset (resize_args[6]) to size each
// tile's dispatch and to zero-size dead trailing tiles after a down-resize.
//
// resize_args = { in, weight_data, stride, padding, dilation, oh_tile,
//                 oh_offset }
void resize_conv2d_gemm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = resize_args.at(0);
  const ValueRef weight_data = resize_args.at(1);
  const ValueRef stride = resize_args.at(2);
  const ValueRef padding = resize_args.at(3);
  const ValueRef dilation = resize_args.at(4);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const size_t ndim = in_sizes.size();
  std::vector<int64_t> new_out_sizes(ndim);

  // N (batch) carries through; C_out = weight_data dim 0.
  new_out_sizes.at(ndim - 4) = in_sizes.at(ndim - 4);
  const std::vector<int64_t> w_sizes = graph->sizes_of(weight_data);
  new_out_sizes.at(ndim - 3) = w_sizes.at(0);

  // Height / Width from the current input, via the shared conv-output helper
  // (same H/W split + formula the direct-conv resize uses). transposed=false,
  // and the args[3] slot (consulted only as an optional ceil_mode) is a
  // non-bool ValueRef, so ceil_mode resolves to false — matching the conv2d
  // semantics.
  const std::vector<int64_t> new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      in_sizes,
      weight_data,
      /*kernel_size_only=*/false,
      {stride, padding, dilation, dilation},
      /*transposed=*/false);
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  graph->virtual_resize(out, new_out_sizes);
}

void add_conv2d_gemm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef im2col_in,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    const ValueRef out,
    const int32_t K_total,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val,
    const int32_t oh_offset,
    const int32_t oh_tile) {
  const int32_t K4_total = K_total / 4;

  // gemm_dims = (K4_total, oh_offset, oh_tile, _unused). All shape-independent:
  // this tile reads scratch rows for oh_tile output-height rows and writes the
  // output rows starting at oh_offset. W_out / H_out are derived in the shader
  // from the refreshed out_sizes UBO (a baked plain-data push constant cannot
  // be updated on resize, but oh_offset / oh_tile never change with shape).
  const utils::ivec4 gemm_dims{K4_total, oh_offset, oh_tile, 0};
  const utils::vec4 clamp_vals{out_min_val, out_max_val, 0.0f, 0.0f};

  // The last two resize_args slots carry the raw oh_tile / oh_offset VALUES,
  // not ValueRef handles: both are build-time constants, so packing the ints
  // directly avoids materializing graph Values for them. The global-wg picker
  // reads them back with static_cast (never get_int) — oh_tile to size the
  // dispatch, oh_offset to zero-size a dead trailing tile after a down-resize.
  // ExecuteNode's resize dirty-tracker treats these slots as value indices: if
  // a packed int happens to collide with a real ValueList index,
  // was_value_updated can RECURSE through toConstValueList() to walk that
  // list's members, so the spurious over-trigger may be a deeper walk than a
  // single lookup. Still memory-safe and read-only (was_value_updated guards
  // out-of-range and in-range idx alike) and benign (the resize recomputes
  // correctly regardless).
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_gemm_shader,
      pick_conv2d_gemm_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{im2col_in, packed_weight, packed_bias}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(out)},
      // Push constants (2 × 16 bytes)
      {PushConstantDataInfo(&gemm_dims, sizeof(gemm_dims)),
       PushConstantDataInfo(&clamp_vals, sizeof(clamp_vals))},
      // Specialization constants
      // activation_type: 0=none, 1=relu, 2=clamp
      {clamp_out ? 2 : 0},
      // Resize args (last two slots = raw oh_tile / oh_offset values, see note
      // above)
      {in,
       weight_data,
       stride,
       padding,
       dilation,
       static_cast<ValueRef>(oh_tile),
       static_cast<ValueRef>(oh_offset)},
      // Resizing logic
      resize_conv2d_gemm_node));
}

} // namespace

//
// Orchestration
//

void conv2d_gemm_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val,
    const std::optional<utils::StorageType> im2col_storage_override) {
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);
  VK_CHECK_COND(in_sizes.size() == 4 && in_sizes[0] == 1);
  VK_CHECK_COND(w_sizes.size() == 4);

  const int64_t C_in = w_sizes[1];
  const int64_t K_h = w_sizes[2];
  const int64_t K_w = w_sizes[3];
  const int64_t H_out = out_sizes[2];
  const int64_t W_out = out_sizes[3];

  const int64_t Cin_padded = utils::align_up_4(C_in);
  const int64_t K_total = K_h * K_w * Cin_padded;
  // Cin_padded is align_up_4(C_in), so K_total is a multiple of 4 and the
  // K4_total = K_total / 4 division below is exact.
  VK_CHECK_COND(K_total % 4 == 0);

  // Extract scalar conv params, scoping the IntListPtrs so they don't keep
  // active value pointers around while we mutate the graph below.
  int32_t stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w;
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    const auto dilation_list = graph.get_int_list(dilation);
    stride_h = utils::safe_downcast<int32_t>(stride_list->at(0));
    stride_w = utils::safe_downcast<int32_t>(stride_list->at(1));
    padding_h = utils::safe_downcast<int32_t>(padding_list->at(0));
    padding_w = utils::safe_downcast<int32_t>(padding_list->at(1));
    dilation_h = utils::safe_downcast<int32_t>(dilation_list->at(0));
    dilation_w = utils::safe_downcast<int32_t>(dilation_list->at(1));
  }

  const int64_t K4_total = K_total / 4;

  // Tile the im2col by output-height rows so the scratch is bounded to the
  // fixed kIm2colScratchBudgetBytes byte budget regardless of resolution. One
  // H-row of im2col is W_out * K_total * elem bytes; oh_tile is the most H-rows
  // whose im2col fits the budget (>= 1), clamped to H_out. The full conv is
  // then materialized in num_tiles = ceil(H_out / oh_tile) tiles. When oh_tile
  // >= H_out this reduces to a single untiled dispatch pair. (Why a fixed
  // num_tiles is safe under dynamic shapes is documented at the dispatch loop
  // below.)
  //
  // Computed BEFORE storage selection because the chosen storage gates on the
  // TILED scratch extent (oh_tile * W_out rows, oh_tile deep), not the full M /
  // H_out. oh_tile depends only on W_out, K_total, and elem_size (the input
  // dtype) — never on the storage type — so there is no circular dependency in
  // ordering it first.
  const int64_t elem_size =
      utils::safe_downcast<int64_t>(vkapi::element_size(graph.dtype_of(in)));
  const int64_t bytes_per_h_row = W_out * K_total * elem_size;
  int64_t oh_tile = kIm2colScratchBudgetBytes / bytes_per_h_row;
  oh_tile = std::max<int64_t>(oh_tile, 1);
  oh_tile = std::min<int64_t>(oh_tile, H_out);
  const int64_t num_tiles = utils::div_up(H_out, oh_tile);

  // The per-tile scratch holds only oh_tile output-height rows, so its extents
  // are M_tile = oh_tile * W_out (texture2d / buffer) or oh_tile-deep
  // (texture3d), not the full M / H_out. With the budget capping oh_tile, the
  // tiled extent rarely exceeds max_texture2d_dim, so texture2d is selected in
  // the common case.

  const int64_t M_tile = oh_tile * W_out;

  // oh_tile reaches the resize fn / wg pickers as the raw int packed into the
  // last resize_args slot (see add_conv2d_*_node) — no materialized graph
  // Value. The scratch's W_out-dependent extent tracks dynamic shapes while
  // oh_tile stays fixed (it is a build-time constant).

  // Pick im2col storage. When an explicit override is provided (test-only),
  // honor it and skip auto-selection. Otherwise run the production
  // auto-selection per device:
  //   - Mali: always buffer (texture sampling on Mali is comparatively slow).
  //   - Others: prefer texture2d (M_tile × K4_total). If the tiled extent
  //     doesn't fit the device's max texture2d dim, fall back to texture3d laid
  //     out as (W_out, oh_tile, K4_total). Buffer is the last-resort fallback.
  utils::StorageType im2col_storage;
  if (im2col_storage_override.has_value()) {
    im2col_storage = im2col_storage_override.value();
    VK_CHECK_COND(
        im2col_storage == utils::kBuffer ||
        im2col_storage == utils::kTexture2D ||
        im2col_storage == utils::kTexture3D);
  } else if (graph.device_is_mali()) {
    im2col_storage = utils::kBuffer;
  } else {
    const uint32_t max_2d = graph.context()->adapter_ptr()->max_texture2d_dim();
    const uint32_t max_3d = graph.context()->adapter_ptr()->max_texture3d_dim();
    const bool fits_2d = utils::safe_downcast<uint32_t>(K4_total) <= max_2d &&
        utils::safe_downcast<uint32_t>(M_tile) <= max_2d;
    const bool fits_3d = utils::safe_downcast<uint32_t>(W_out) <= max_3d &&
        utils::safe_downcast<uint32_t>(oh_tile) <= max_3d &&
        utils::safe_downcast<uint32_t>(K4_total) <= max_3d;
    if (fits_2d) {
      im2col_storage = utils::kTexture2D;
    } else if (fits_3d) {
      im2col_storage = utils::kTexture3D;
    } else {
      im2col_storage = utils::kBuffer;
    }
  }

  // Allocate ONE im2col scratch tensor sized for a single tile (oh_tile rows),
  // reused across all tiles. The im2col value is produced by each tile's im2col
  // node and consumed immediately by that tile's GEMM node; reusing the same
  // TmpTensor across tiles serializes them via the backend's automatic
  // read/write barriers (tile t's GEMM finishes reading before tile t+1's
  // im2col overwrites). Using a TmpTensor also lets the memory planner alias
  // one backing buffer across the non-overlapping im2col lifetimes of every
  // conv2d layer, so peak memory tracks the largest single tile's scratch (<=
  // budget) rather than the sum. The TmpTensor must outlive the last GEMM node,
  // so it lives to the end of this function.
  //
  // The 2D and buffer variants use a flat [oh_tile * W_out, K_total]
  // kWidthPacked shape; the texture3d variant uses [1, K_total, oh_tile, W_out]
  // kChannelsPacked so K4 lays along Z. Hoist the per-storage differences into
  // locals so the TmpTensor is constructed exactly once and never copied/moved.
  std::vector<int64_t> im2col_sizes;
  utils::StorageType im2col_tmp_storage;
  utils::GPUMemoryLayout im2col_layout;
  if (im2col_storage == utils::kTexture3D) {
    im2col_sizes = {1, K_total, oh_tile, W_out};
    im2col_tmp_storage = utils::kTexture3D;
    im2col_layout = utils::kChannelsPacked;
  } else {
    im2col_sizes = {oh_tile * W_out, K_total};
    im2col_tmp_storage = im2col_storage;
    im2col_layout = utils::kWidthPacked;
  }
  TmpTensor im2col_tmp(
      &graph,
      im2col_sizes,
      graph.dtype_of(in),
      im2col_tmp_storage,
      im2col_layout);
  const ValueRef im2col_tensor = im2col_tmp.vref;

  // Prepack weight for the GEMM directly from the serialized
  // [C_out, C_in, K_h, K_w] weight on the GPU (shared across all tiles). The
  // serialized data is read as-is (never CPU-repacked); the prepack shader does
  // the im2col K-axis reorder + 4x4 transpose into the layout conv2d_gemm.glsl
  // loads via load_packed_weight_tile_with_checks.
  ValueRef packed_weight = prepack_conv2d_gemm_weight(graph, weight_data);

  // Bias prepack: matches the bias format conv2d_gemm expects. prepack_biases
  // only reads dim 0 (= C_out) of the weight, so the original 4D weight works
  // directly.
  ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      /*transposed=*/false,
      utils::kTexture2D,
      utils::kWidthPacked);

  check_conv_args(graph, in, out);

  // Emit one (im2col -> GEMM) dispatch pair per tile, interleaved so each
  // tile's GEMM reads the scratch its im2col just wrote before the next tile
  // overwrites it.
  //
  // num_tiles (and oh_tile) are fixed here at graph-build time: the per-tile
  // dispatch count must be static (DynamicDispatchNode does not add/remove
  // nodes on resize). This is correct ONLY because ET-VK builds these tensors
  // at the dynamic UPPER BOUND, so trigger_resize() can only shrink H_out/W_out
  // (runtime <= build-time). Three consequences make the fixed tiling safe:
  //   - num_tiles, from the build-time (max) H_out, always covers the runtime
  //     row count;
  //   - a smaller runtime H_out just leaves trailing tiles (oh_offset >= the
  //     current H_out) with no live output rows; the GEMM global-wg picker
  //     zero-sizes the dispatch for such a tile (DispatchNode::encode() skips a
  //     0-component global wg), so a dead trailing tile costs no GEMM work
  //     after a down-resize. (Its im2col node still dispatches — a cheap
  //     per-thread gather that writes zeros into the unused scratch rows; the
  //     static node-count constraint means the im2col node cannot be removed,
  //     only the dominant GEMM work is elided.)
  //   - the scratch, sized from the build-time (max) shape, is an upper bound,
  //     so memory stays capped with no reallocation on resize.
  // Load-bearing assumption: if a runtime shape ever EXCEEDED the build-time
  // bound, this fixed num_tiles would under-cover and silently drop the extra
  // output rows. ET-VK's upper-bound build guarantees this cannot happen; a
  // future change that breaks the upper-bound invariant would need a runtime
  // tile-count guard here.
  for (int64_t t = 0; t < num_tiles; ++t) {
    const int32_t oh_offset = utils::safe_downcast<int32_t>(t * oh_tile);
    const int32_t oh_tile_i32 = utils::safe_downcast<int32_t>(oh_tile);

    add_conv2d_im2col_node(
        graph,
        in,
        im2col_tensor,
        weight_data,
        stride,
        padding,
        dilation,
        utils::safe_downcast<int32_t>(K_h),
        utils::safe_downcast<int32_t>(K_w),
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        utils::safe_downcast<int32_t>(Cin_padded),
        oh_offset,
        oh_tile_i32);

    add_conv2d_gemm_node(
        graph,
        in,
        weight_data,
        stride,
        padding,
        dilation,
        im2col_tensor,
        packed_weight,
        packed_bias,
        out,
        utils::safe_downcast<int32_t>(K_total),
        clamp_out,
        out_min_val,
        out_max_val,
        oh_offset,
        oh_tile_i32);
  }
}

//
// Op registration — matches aten.convolution.default's 10-arg signature:
//   in, weight, bias, stride, padding, dilation, transposed,
//   output_padding, groups, out
//
// Only the conv2d non-transposed, groups=1 case is supported.

void conv2d_gemm_op(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  VK_CHECK_COND(args.size() == 10);
  const ValueRef in = args[0];
  const ValueRef weight = args[1];
  const ValueRef bias = args[2];
  const ValueRef stride = args[3];
  const ValueRef padding = args[4];
  const ValueRef dilation = args[5];
  const ValueRef transposed = args[6];
  const ValueRef /*output_padding*/ _output_padding = args[7];
  (void)_output_padding;
  const ValueRef groups = args[8];
  const ValueRef out = args[9];

  VK_CHECK_COND(graph.get_bool(transposed) == false);
  VK_CHECK_COND(graph.get_int(groups) == 1);

  conv2d_gemm_impl(
      graph,
      in,
      weight,
      bias,
      stride,
      padding,
      dilation,
      out,
      /*clamp_out=*/false,
      /*out_min_val=*/0.0f,
      /*out_max_val=*/0.0f);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv2d_gemm.default, conv2d_gemm_op);
}

} // namespace vkcompute
