/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dDirect.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <array>

namespace vkcompute {

vkapi::ShaderInfo get_conv2d_prepack_shader(
    ComputeGraph& graph,
    const ValueRef out,
    const Conv2dMethod method,
    const ValueRef weight) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);
  switch (method) {
    case Conv2dMethod::Depthwise:
      kernel_name = "conv2d_dw";
      break;
    case Conv2dMethod::Pointwise:
    case Conv2dMethod::SlidingWindow:
      kernel_name = "conv2d";
      break;
    case Conv2dMethod::Transposed:
      kernel_name = "conv_transpose2d";
      break;
  }
  kernel_name += "_prepack_weights";
  if (graph.is_buffer_storage(out)) {
    kernel_name += "_buffer";
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_dtype_suffix(kernel_name, graph.get_staging_dtype_for(weight));

  return VK_KERNEL_FROM_STR(kernel_name);
}

std::vector<int64_t> get_final_sizes(
    const std::vector<int64_t>& original_sizes,
    const Conv2dMethod method) {
  int64_t batch_padded = utils::align_up_4(utils::val_at(-4, original_sizes));
  if (method == Conv2dMethod::SlidingWindow) {
    batch_padded = utils::align_up(batch_padded, int64_t(8));
  }
  int64_t channels_padded =
      utils::align_up_4(utils::val_at(-3, original_sizes));
  int64_t height = utils::val_at(-2, original_sizes);
  int64_t width = utils::val_at(-1, original_sizes);

  switch (method) {
    case Conv2dMethod::Depthwise:
      return std::vector<int64_t>{4, batch_padded / 4, height * width};
    case Conv2dMethod::Pointwise:
    case Conv2dMethod::SlidingWindow:
      return std::vector<int64_t>{
          4, batch_padded * height / 4, channels_padded * width};
    case Conv2dMethod::Transposed:
      return std::vector<int64_t>{
          4, channels_padded * height / 4, batch_padded * width};
  }
}

ValueRef prepack_weights(
    ComputeGraph& graph,
    const ValueRef vref,
    const Conv2dMethod method,
    const utils::StorageType storage_type = utils::kTexture2D) {
  const auto original_sizes = graph.sizes_of(vref);
  const auto final_sizes = get_final_sizes(original_sizes, method);

  if (storage_type == utils::kBuffer) {
    VK_CHECK_COND(graph.dtype_of(vref) == vkapi::kHalf);
    const auto width = utils::safe_downcast<uint64_t>(final_sizes.at(2));
    const auto height = utils::safe_downcast<uint64_t>(final_sizes.at(1));
    const auto byte_limit =
        utils::safe_downcast<uint64_t>(graph.max_buffer_numel());
    VK_CHECK_COND(
        width > 0 && height > 0 && width <= byte_limit / 8 / height,
        "Packed Half weight buffer exceeds byte range");
  }

  ValueRef v = graph.add_tensor(
      final_sizes, graph.dtype_of(vref), storage_type, utils::kChannelsPacked);

  vkapi::ShaderInfo shader = get_conv2d_prepack_shader(graph, v, method, vref);
  const GlobalWorkGrid gwg = storage_type == utils::kBuffer
      ? GlobalWorkGrid(
            {utils::safe_downcast<uint32_t>(final_sizes.at(2)),
             utils::safe_downcast<uint32_t>(final_sizes.at(1)),
             1u},
            kExplicitWorkGrid)
      : graph.create_gwg(v);
  const auto original_sizes_pc =
      utils::make_ivec4(original_sizes, /*reverse = */ true);
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      gwg,
      graph.create_lwg(gwg),
      vref,
      v,
      {},
      // Specialization constants
      {graph.packed_dim_of(v)},
      {graph.sizes_pc_of(v),
       PushConstantDataInfo(&original_sizes_pc, sizeof(original_sizes_pc))}));

  return v;
}

namespace {

struct Conv2dParams final {
  utils::ivec2 overlay_region;
  int in_group_size;
};

} // namespace

struct OutputParams final {
  float out_min;
  float out_max;
};

Conv2dParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const Kernel2dParams& p,
    const bool transposed) {
  const auto& overlay_region = utils::make_ivec2({
      p.kernel_size[0] + (p.kernel_size[0] - 1) * (p.dilation[0] - 1),
      p.kernel_size[1] + (p.kernel_size[1] - 1) * (p.dilation[1] - 1),
  });
  const auto weight_sizes = graph.sizes_of(weight);
  const int32_t in_group_size = utils::safe_downcast<int32_t>(
      utils::align_up_4(transposed ? weight_sizes.at(0) : weight_sizes.at(1)));
  return {overlay_region, in_group_size};
}

void check_conv2d_params(const Kernel2dParams& p, const bool transposed) {
  if (transposed) {
    if (p.dilation[0] > 1 || p.dilation[1] > 1) {
      VK_THROW(
          "aten.convolution.default: transposed = true, dilation > 1 is not supported yet!");
    }
  }
}

namespace {

constexpr uint64_t kMinTileInvocations = 1024;
constexpr uint64_t kMinNarrowTileInvocations = 8192;
constexpr uint64_t kMinTunedInvocations = 65536;

enum class Conv2dShader {
  TextureBorder8,
  Buffer4,
  BufferOffset8,
  Texture4,
  TextureBorder4,
  TextureRange4,
  TextureRow4,
  TextureRowBorder4,
  BufferBorder8,
  BufferRange4,
  Gather4,
  Transposed,
};

struct Conv2dVariant {
  const char* name;
  std::array<uint32_t, 3> tile;
  bool full_grid;
  bool tune_channels;
  bool transposed = false;
};

constexpr std::array<Conv2dVariant, 12> kConv2dVariants{{
    {"conv2d_2x2x8_border", {2, 2, 2}, false, true},
    {"conv2d_2x2x4_buffer", {2, 2, 1}, false, false},
    {"conv2d_2x2x8_border_buffer_offset", {2, 2, 2}, false, true},
    {"conv2d_2x2x4", {2, 2, 1}, false, false},
    {"conv2d_2x2x4_border", {2, 2, 1}, false, false},
    {"conv2d_1x1x4", {1, 1, 1}, false, false},
    {"conv2d_2x1x4", {2, 1, 1}, false, false},
    {"conv2d_2x1x4_border", {2, 1, 1}, false, false},
    {"conv2d_2x2x8_border_buffer", {2, 2, 2}, false, false},
    {"conv2d_1x1x4_buffer", {1, 1, 1}, false, false},
    {"conv2d_2x2x4_grouped_gather", {2, 2, 1}, false, false},
    {"conv_transpose2d", {1, 1, 1}, true, false, true},
}};

/*
 * Picks the shader variant of a direct convolution, which also determines its
 * output tile, grid and local size. Unless noted otherwise, each rule comes
 * from timing the candidate variants against each other on Adreno, Mali,
 * Xclipse (AMD) and PowerVR phones, over a benchmark set of small to medium
 * direct convolutions (varied channels, groups, strides, kernels and single-row
 * outputs) and large Half 3x3 / 5x5 convolutions. In order:
 *
 * 1. Transposed convolutions use the transposed shader.
 * 2. Groups whose input or output channels per group are not a multiple of 4
 *    use the per-channel gather variant. This is required for correctness: the
 *    other variants assume every texel belongs to a single group.
 * 3. If 2x2 tiles with two output slices would launch too few invocations to
 *    fill the GPU, a 1x1 tile is used. The threshold is higher for grouped Half
 *    convolutions with at most 8 output channels per group on Adreno, and for
 *    irregular shapes on AMD (grouped, output channels per group not a multiple
 *    of 8, or non-unit stride or dilation), except grouped unit-stride ones
 *    with more than 8 output channels per group. Both thresholds were placed
 *    between the measured shapes that were faster with 1x1 and 2x2 tiles; the
 *    sizes in between were not swept.
 * 4. Otherwise, padding taps are read as zeros through the sampler border when
 *    the input texture allows it, which avoids per-tap bounds checks. If not,
 *    they are bounds-checked and one output slice is computed:
 *    - Buffer weights (Mali Half) and ungrouped Mali Half use a 2x2 tile with
 *      two output slices; buffer weights with a square unit-stride kernel use
 *      constant texel offsets. Both were tuned on the large Half convolutions,
 *      where buffer weights cut latency by about 40% versus texture weights.
 *    - Single-row outputs use a 2x1 tile, which skips the unused second row.
 *    - Everything else uses a 2x2 tile with two output slices, or one if the
 *      output channels per group are not a multiple of 8 (avoids padded work
 *      in the second slice), for Float on Mali (two slices spill registers in
 *      the Mali offline compiler; not timed), or for grouped Half on Adreno and
 *      irregular shapes on AMD (two slices measured slower).
 *
 * Mali Half 2x2 tiles dispatch one invocation per output texel, with the
 * excess exiting immediately. That is the dispatch the Mali Half variants were
 * tuned with, and it measured up to 5x faster than a direct shader computing
 * one output texel per invocation. The tiled grid, which measured faster on the
 * other GPUs, was not timed on Mali Half.
 *
 * Fast tiled convolutions computing two output slices per invocation use tuned
 * local sizes on Mali Half, and on other GPUs when the tiled grid has at least
 * kMinTunedInvocations invocations. On the other GPUs, tuned sizes measured up
 * to about 2.5x faster on large planes (128 output channels at 189x252) and up
 * to about 2x slower on smaller grids. The threshold was placed between the
 * largest grid measured slower (27648 invocations) and the smallest measured
 * faster (191520); the sizes in between were not swept.
 */
Conv2dVariant pick_conv2d_variant(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (graph->get_bool(resize_args.at(4))) {
    return kConv2dVariants[static_cast<size_t>(Conv2dShader::Transposed)];
  }
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef weight = args.at(1).refs.at(1);
  const ValueRef weight_data = resize_args.at(0);
  const int64_t groups = graph->get_int(resize_args.at(8));
  const int64_t in_group = graph->size_at<int64_t>(1, weight_data);
  const int64_t out_group = graph->size_at<int64_t>(0, weight_data) / groups;
  const int32_t out_slices = graph->logical_limits_of(out)[2];
  const int64_t height = graph->size_at<int64_t>(-2, out);
  const int64_t width = graph->size_at<int64_t>(-1, out);
  const bool fast_tiled = graph->get_bool(resize_args.at(9));
  const bool grouped = groups > 1;
  const bool half = graph->dtype_of(out) == vkapi::kHalf;
  const bool mali = graph->device_is_mali();
  const bool buffer = graph->is_buffer_storage(weight);
  const bool even_slices = out_slices > 0 && out_slices % 2 == 0;
  const uint64_t invocations = utils::div_up(uint64_t(width), uint64_t(2)) *
      utils::div_up(uint64_t(height), uint64_t(2)) *
      utils::div_up(uint64_t(out_slices), uint64_t(2));
  Conv2dShader shader;

  if (grouped && (in_group % 4 != 0 || out_group % 4 != 0)) {
    shader = Conv2dShader::Gather4;
  } else {
    const auto& stride = *graph->get_int_list(resize_args.at(1));
    const auto& dilation = *graph->get_int_list(resize_args.at(3));
    const bool unit_sampling = stride[0] == 1 && stride[1] == 1 &&
        dilation[0] == 1 && dilation[1] == 1;
    const bool amd = graph->device_is_amd();
    const bool irregular = grouped || out_group % 8 != 0 || !unit_sampling;
    const uint64_t minimum =
        (amd && irregular && !(grouped && unit_sampling && out_group > 8)) ||
            (half && grouped && out_group <= 8 && graph->device_is_adreno())
        ? kMinNarrowTileInvocations
        : kMinTileInvocations;
    // Smaller tiles expose more invocations when a wide tile would underfill
    // the GPU.
    if (invocations < minimum) {
      shader =
          buffer ? Conv2dShader::BufferRange4 : Conv2dShader::TextureRange4;
    } else {
      const bool border = !graph->get_bool(resize_args.at(6)) &&
          graph->samples_zero_outside_tensor(in);
      if (buffer) {
        const auto sizes = graph->sizes_of(weight_data);
        const bool offsets = unit_sampling && sizes[2] == sizes[3];
        shader = !border ? Conv2dShader::Buffer4
            : offsets    ? Conv2dShader::BufferOffset8
                         : Conv2dShader::BufferBorder8;
      } else if (half && mali && !grouped) {
        shader = border ? Conv2dShader::TextureBorder8 : Conv2dShader::Texture4;
      } else if (height == 1) {
        shader = border ? Conv2dShader::TextureRowBorder4
                        : Conv2dShader::TextureRow4;
      } else {
        const bool one_slice = !border || out_group % 8 != 0 ||
            (half && grouped && graph->device_is_adreno()) || (!half && mali) ||
            (amd && irregular);
        shader = !border ? Conv2dShader::Texture4
            : one_slice  ? Conv2dShader::TextureBorder4
                         : Conv2dShader::TextureBorder8;
      }
    }
  }

  Conv2dVariant variant = kConv2dVariants[static_cast<size_t>(shader)];
  const bool mali_grid =
      half && mali && variant.tile[0] == 2 && variant.tile[1] == 2;
  variant.full_grid = mali_grid;
  variant.tune_channels = variant.tune_channels &&
      (shader != Conv2dShader::TextureBorder8 || even_slices) && fast_tiled &&
      (mali_grid || invocations >= kMinTunedInvocations);
  return variant;
}

vkapi::ShaderInfo pick_conv2d_tiled_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const Conv2dVariant variant = pick_conv2d_variant(graph, args, resize_args);
  std::string name = variant.name;
  if (variant.transposed && graph->get_bool(resize_args.at(7))) {
    name += "_clamp";
  }
  add_dtype_suffix(name, graph->dtype_of(args.at(0).refs.at(0)));
  return VK_KERNEL_FROM_STR(name);
}

GlobalWorkGrid conv2d_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const Conv2dVariant variant = pick_conv2d_variant(graph, args, resize_args);
  const GlobalWorkGrid grid = graph->create_gwg(args.at(0).refs.at(0));
  if (variant.full_grid) {
    return grid;
  }
  return GlobalWorkGrid(
      {utils::div_up(grid[0], variant.tile[0]),
       utils::div_up(grid[1], variant.tile[1]),
       utils::div_up(grid[2], variant.tile[2])},
      kTiledWorkGrid);
}

/*
 * Tuned dispatches size the workgroup by GPU family and grid depth (output
 * slices): Adreno uses 16x1x16 from a depth of 32 and 8x1x16 from 16, other
 * GPUs use 8x1x8 from 16, and shallower grids use 8x2x4. All other dispatches
 * use the default local size for their grid.
 *
 * The sizes come from sweeping 12 local sizes over large Half 3x3 / 5x5
 * convolutions (32 to 128 output channels, one invocation per output texel) on
 * Adreno, Mali, Xclipse and PowerVR phones. Deeper workgroups were faster for
 * outputs with more slices, especially on Adreno. The rule is the simplest one
 * by GPU family and depth that was never more than about 2% slower than 8x2x4
 * on a measured shape; it does not always pick the fastest size.
 */
LocalWorkGroup conv2d_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const Conv2dVariant variant = pick_conv2d_variant(graph, args, resize_args);
  if (variant.tune_channels) {
    const uint32_t slices = gwg[2];
    const bool adreno = graph->device_is_adreno();
    if (slices >= 32u && adreno) {
      return LocalWorkGroup(16u, 1u, 16u, 256u);
    }
    if (slices >= 16u) {
      return adreno ? LocalWorkGroup(8u, 1u, 16u, 128u)
                    : LocalWorkGroup(8u, 1u, 8u);
    }
    return LocalWorkGroup(8u, 2u, 4u);
  }
  return graph->create_lwg(gwg);
}

} // namespace

/*
 * Fast tiled convolutions are the ones the tiled shaders are tuned for:
 * non-transposed, ungrouped, batch-1 convolutions on Texture3D tensors with a
 * square 3x3 or 5x5 kernel, unit stride and dilation, and "same" padding. They
 * skip im2col on large output planes and may use tuned local sizes.
 *
 * The class is the geometry of the large Half convolutions the shaders and
 * local sizes were tuned on, and of the shapes used to place the im2col cutoff.
 * The rule accepts any dtype, although Float convolutions were not part of that
 * tuning.
 */
bool can_use_fast_tiled_conv2d(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups) {
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const int32_t kernel = kernel_params.kernel_size[0];
  return !transposed && groups == 1 && in_sizes.size() == 4 &&
      in_sizes[0] == 1 && graph.storage_type_of(in) == utils::kTexture3D &&
      graph.storage_type_of(out) == utils::kTexture3D &&
      (kernel == 3 || kernel == 5) && kernel_params.kernel_size[1] == kernel &&
      kernel_params.stride[0] == 1 && kernel_params.stride[1] == 1 &&
      kernel_params.dilation[0] == 1 && kernel_params.dilation[1] == 1 &&
      kernel_params.padding[0] == kernel / 2 &&
      kernel_params.padding[1] == kernel / 2;
}

bool can_use_direct_conv2d(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& p,
    const int64_t groups) {
  const auto in_sizes = graph.sizes_of(in);
  const auto out_sizes = graph.sizes_of(out);
  const auto weight_sizes = graph.sizes_of(weight_data);
  const auto dtype = graph.dtype_of(in);
  return in_sizes.size() == 4 && out_sizes.size() == 4 &&
      weight_sizes.size() == 4 && in_sizes[0] == 1 && out_sizes[0] == 1 &&
      (dtype == vkapi::kFloat || dtype == vkapi::kHalf) &&
      graph.dtype_of(out) == dtype &&
      (graph.dtype_of(weight_data) == vkapi::kFloat ||
       graph.dtype_of(weight_data) == vkapi::kHalf) &&
      graph.storage_type_of(in) == utils::kTexture3D &&
      graph.storage_type_of(out) == utils::kTexture3D && groups > 0 &&
      in_sizes[1] > 0 && weight_sizes[0] > 0 && in_sizes[1] % groups == 0 &&
      weight_sizes[0] % groups == 0 &&
      in_sizes[1] / groups == weight_sizes[1] &&
      out_sizes[1] == weight_sizes[0] && p.kernel_size[0] > 0 &&
      p.kernel_size[1] > 0 && p.stride[0] > 0 && p.stride[1] > 0 &&
      p.dilation[0] > 0 && p.dilation[1] > 0 && p.padding[0] >= 0 &&
      p.padding[1] >= 0;
}

void add_conv2d_direct_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const Conv2dParams& extra_params,
    const OutputParams& out_params,
    const ValueRef groups,
    const int64_t groups_val,
    const bool has_bias,
    const bool force_masked,
    const bool fast_tiled,
    const bool clamp_out) {
  vkapi::ParamsBindList param_buffers = {
      graph.logical_limits_ubo(out),
      graph.sizes_ubo(in),
      graph.create_params_buffer(kernel_params),
      graph.create_params_buffer(extra_params),
      graph.create_params_buffer(out_params),
  };

  const int32_t in_quads = extra_params.in_group_size / 4;
  vkapi::SpecVarList spec_vars = {utils::safe_downcast<int32_t>(groups_val)};
  if (!graph.get_bool(transposed)) {
    spec_vars.append(vkapi::SpecVar(in_quads));
    spec_vars.append(vkapi::SpecVar(kernel_params.kernel_size[0]));
    spec_vars.append(vkapi::SpecVar(clamp_out));
    spec_vars.append(vkapi::SpecVar(has_bias));
    spec_vars.append(vkapi::SpecVar(graph.size_at<int32_t>(1, weight_data)));
    spec_vars.append(vkapi::SpecVar(utils::safe_downcast<int32_t>(
        graph.size_at<int64_t>(0, weight_data) / groups_val)));
  }

  const std::vector<ArgGroup> args = {
      {out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}};
  const std::vector<ValueRef> resize_args = {
      weight_data,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      graph.add_scalar<bool>(force_masked),
      graph.add_scalar<bool>(clamp_out),
      groups,
      graph.add_scalar<bool>(fast_tiled)};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_conv2d_tiled_shader,
      conv2d_gwg,
      conv2d_lwg,
      args,
      param_buffers,
      {},
      spec_vars,
      resize_args,
      resize_conv2d_node));
}

void conv2d_direct_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef groups,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val,
    const bool force_masked) {
  const bool transposed_val = graph.get_bool(transposed);
  const int64_t groups_val = graph.get_int(groups);
  const Conv2dMethod method =
      transposed_val ? Conv2dMethod::Transposed : Conv2dMethod::SlidingWindow;

  const Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      weight_data,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);

  const int32_t out_quads = graph.logical_limits_of(out)[2];
  const bool buffer_weights = !transposed_val && groups_val == 1 &&
      graph.dtype_of(in) == vkapi::kHalf &&
      graph.dtype_of(weight_data) == vkapi::kHalf && graph.device_is_mali() &&
      graph.float16_buffers_enabled() && out_quads > 0 && out_quads % 2 == 0;
  ValueRef arg_weight = prepack_weights(
      graph,
      weight_data,
      method,
      buffer_weights ? utils::kBuffer : utils::kTexture2D);
  ValueRef arg_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      transposed_val,
      /* storage_type = */ utils::kTexture2D,
      /* memory_layout = */ utils::kWidthPacked);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  if (in_sizes.at(0) > 1) {
    VK_THROW("conv2d: input batch size > 1 is not supported yet!");
  }

  check_conv_args(graph, in, out);

  Conv2dParams extra_params =
      create_conv2d_params(graph, weight_data, kernel_params, transposed_val);

  OutputParams out_params = {out_min_val, out_max_val};

  check_conv2d_params(kernel_params, transposed_val);
  if (!transposed_val) {
    VK_CHECK_COND(
        can_use_direct_conv2d(
            graph, in, weight_data, out, kernel_params, groups_val),
        "Direct convolution requires batch-1 channels-packed Texture3D activations and valid channels/groups");
  }
  const bool fast_tiled = can_use_fast_tiled_conv2d(
      graph, in, out, kernel_params, transposed_val, groups_val);

  add_conv2d_direct_node(
      graph,
      in,
      arg_weight,
      arg_bias,
      weight_data,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      out,
      kernel_params,
      extra_params,
      out_params,
      groups,
      groups_val,
      !graph.val_is_none(bias),
      force_masked,
      fast_tiled,
      clamp_out);
}

} // namespace vkcompute
