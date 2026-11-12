/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dDirect.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/im2col/Conv2dGemm.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

Conv2dEpilogueArgs::Conv2dEpilogueArgs(
    const Conv2dEpilogue& epilogue,
    const ValueRef in,
    const ValueRef packed_bias,
    const ValueRef packed_scale)
    : residual(epilogue.has_residual() ? epilogue.residual : in),
      scale(epilogue.has_scale() ? packed_scale : packed_bias) {}

ValueRef prepack_conv2d_residual_scale(
    ComputeGraph& graph,
    const Conv2dEpilogue& epilogue,
    const ValueRef weight_data,
    const bool transposed) {
  if (!epilogue.has_scale()) {
    return kDummyValueRef;
  }
  return prepack_biases(
      graph,
      epilogue.scale,
      weight_data,
      transposed,
      utils::kTexture2D,
      utils::kWidthPacked);
}

void check_conv2d_residual_on_resize(
    ComputeGraph* graph,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef residual = resize_args.back();
  VK_CHECK_COND(
      !is_valid(residual) || graph->sizes_of(residual) == graph->sizes_of(out),
      "conv2d residual must match the resized output's sizes");
}

Conv2dEpilogueParams create_conv2d_epilogue_params(
    const Conv2dEpilogue& epilogue,
    const bool f16_epilogue) {
  if (!f16_epilogue) {
    return {
        epilogue.activation_a, epilogue.activation_b, epilogue.residual_alpha};
  }
  const auto round_to_half = [](const float f) {
    return api::half_to_float(api::float_to_half(f));
  };
  return {
      round_to_half(epilogue.activation_a),
      round_to_half(epilogue.activation_b),
      round_to_half(epilogue.residual_alpha)};
}

void append_conv2d_epilogue_spec_vars(
    vkapi::SpecVarList& spec_vars,
    const Conv2dEpilogue& epilogue) {
  // The shaders ignore activation_op without an activation.
  const UnaryOpType activation =
      epilogue.activation.value_or(UnaryOpType::kClamp);
  spec_vars.append(vkapi::SpecVar(epilogue.has_activation()));
  spec_vars.append(vkapi::SpecVar(static_cast<int32_t>(activation)));
  spec_vars.append(vkapi::SpecVar(epilogue.has_residual()));
  spec_vars.append(vkapi::SpecVar(epilogue.has_scale()));
  spec_vars.append(vkapi::SpecVar(static_cast<int32_t>(epilogue.residual_op)));
}

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  size_t ndim = graph->dim_of(self);
  std::vector<int64_t> new_out_sizes(ndim);
  const bool transposed = graph->get_bool(extra_args.at(4));

  std::vector<int64_t> self_sizes = graph->sizes_of(self);
  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self_sizes.at(ndim - 4);
  }

  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));
  const auto& weight_sizes = weight_ref->sizes;
  new_out_sizes.at(ndim - 3) =
      transposed ? weight_sizes.at(ndim - 3) : weight_sizes.at(ndim - 4);

  // Height, Width
  const auto& new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self_sizes,
      extra_args.at(0),
      /*kernel_size_only = */ false,
      {extra_args.at(1), extra_args.at(2), extra_args.at(3), extra_args.at(5)},
      transposed);
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  graph->virtual_resize(out, new_out_sizes);
  check_conv2d_residual_on_resize(graph, out, extra_args);
}

Conv2dMethod get_conv2d_method(
    ComputeGraph& graph,
    const ValueRef weight,
    const int64_t groups,
    const bool transposed) {
  const auto weight_sizes = graph.sizes_of(weight);
  if (!transposed && weight_sizes.at(0) == groups && weight_sizes.at(1) == 1) {
    return Conv2dMethod::Depthwise;
  }
  if (transposed) {
    return Conv2dMethod::Transposed;
  }
  if (weight_sizes.at(2) == 1 && weight_sizes.at(3) == 1) {
    return Conv2dMethod::Pointwise;
  }
  return Conv2dMethod::SlidingWindow;
}

// Decide whether a SlidingWindow conv2d should be computed via the
// im2col + GEMM path (conv2d_gemm_impl) instead of the direct convolution
// shader. Across 26 configs on Mali-G715 (buffer path) and Adreno SM8650
// (texture path): FP32 cases were numerically verified against the reference;
// FP16 cases were routing/dispatch-validated only (the reference is float-only
// for the large shapes, so FP16 outputs were not numerically checked).
//
// Only called for SlidingWindow conv2d (1x1 is routed to conv2d_pw and
// Depthwise/Transposed are handled before the call site).
//
// Preconditions (fall back to direct conv if any fail — the im2col path is
// either not applicable or not beneficial):
//   - groups == 1
//   - dilation == 1 (all dims)
//
// Selection rule: use im2col once the output channel count is large enough to
// amortize the fixed ~N*K_total im2col gather cost.
constexpr int64_t kIm2colMinCOut = 128;

// A cheap gather is a second, independent reason to take im2col. The gather
// materializes an N x K_total matrix before the GEMM runs, and a large gather
// is fine when c_out is large because the GEMM reads it back c_out times. When
// c_out is small it is only worth paying if the matrix is small outright, which
// is where the direct shader on Mali loses badly: an 80x80 3x3 conv wants ~4M
// elements, a 640x640 9x9 conv wants ~100M.
constexpr int64_t kIm2colMaxCheapGatherElements = 32 * 1024 * 1024;

// Larger output planes favor the tiled direct shaders over im2col + GEMM. From
// timing 29 Half 3x3 / 5x5 shapes on Adreno, Mali, Xclipse and PowerVR phones:
// above this size, tiled was faster on all but two noisy PowerVR shapes; below
// about 1200 output pixels, im2col was up to 1.55x faster.
constexpr int64_t kTiledMinOutPixels = 1536;

bool should_use_conv2d_im2col(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const int64_t groups_val,
    const Kernel2dParams& kernel_params,
    const ValueRef out,
    const bool using_fast_tiled_direct_conv2d) {
  if (groups_val != 1) {
    return false;
  }
  if (kernel_params.dilation[0] != 1 || kernel_params.dilation[1] != 1) {
    return false;
  }
  const auto weight_sizes = graph.sizes_of(weight_data);
  const int64_t c_out = weight_sizes.at(0);
  const int64_t k_total =
      weight_sizes.at(1) * weight_sizes.at(2) * weight_sizes.at(3);

  const auto out_sizes = graph.sizes_of(out);
  const size_t ndim = out_sizes.size();
  const int64_t n = out_sizes.at(ndim - 1) * out_sizes.at(ndim - 2);
  if (using_fast_tiled_direct_conv2d && n >= kTiledMinOutPixels) {
    return false;
  }
  return c_out >= kIm2colMinCOut ||
      (graph.device_is_mali() && n * k_total <= kIm2colMaxCheapGatherElements);
}

namespace {

// The conv2d shader epilogue implements only these residual ops and
// activations.
BinaryOpType residual_op_from_string(const std::string& name) {
  if (name == "add") {
    return BinaryOpType::kAdd;
  }
  if (name == "sub") {
    return BinaryOpType::kSub;
  }
  if (name == "mul") {
    return BinaryOpType::kMul;
  }
  VK_THROW("Unsupported conv2d residual op: ", name);
}

UnaryOpType activation_from_string(const std::string& name) {
  if (name == "relu") {
    return UnaryOpType::kRelu;
  }
  if (name == "clamp") {
    return UnaryOpType::kClamp;
  }
  if (name == "leaky_relu") {
    return UnaryOpType::kLeakyRelu;
  }
  VK_THROW("Unsupported conv2d activation: ", name);
}

void check_conv2d_epilogue(
    ComputeGraph& graph,
    const Conv2dEpilogue& epilogue,
    const ValueRef weight_data,
    const ValueRef out) {
  if (epilogue.has_residual()) {
    const ValueRef residual = epilogue.residual;
    VK_CHECK_COND(
        graph.sizes_of(residual) == graph.sizes_of(out) &&
            graph.dtype_of(residual) == graph.dtype_of(out) &&
            graph.storage_type_of(residual) == utils::kTexture3D &&
            graph.hashed_layout_of(residual) == graph.hashed_layout_of(out),
        "conv2d residual must match the output's sizes, dtype and layout");
  }
  if (epilogue.has_scale()) {
    VK_CHECK_COND(
        epilogue.has_residual(), "conv2d residual scale requires a residual");
    VK_CHECK_COND(
        graph.val_is_tref(epilogue.scale),
        "conv2d residual scale must be a constant");
    // Other shapes with one value per channel, such as [C], broadcast along
    // another dim of the output.
    const std::vector<int64_t> sizes = graph.sizes_of(epilogue.scale);
    const int64_t channels = graph.size_at<int64_t>(-3, out);
    VK_CHECK_COND(
        (sizes.size() == 3 || sizes.size() == 4) &&
            sizes.at(sizes.size() - 3) == channels &&
            utils::multiply_integers(sizes) == channels &&
            graph.dtype_of(epilogue.scale) == graph.dtype_of(weight_data),
        "conv2d residual scale must be a [1, C, 1, 1] or [C, 1, 1] tensor "
        "with the weight's dtype");
  }
}

} // namespace

Conv2dEpilogue conv2d_epilogue_from_args(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  Conv2dEpilogue epilogue;
  if (args.size() == 12) {
    // conv_with_clamp passes out_min and out_max before the output
    epilogue.activation = UnaryOpType::kClamp;
    epilogue.activation_a = graph.extract_scalar<float>(args[9]);
    epilogue.activation_b = graph.extract_scalar<float>(args[10]);
    return epilogue;
  }
  if (args.size() == 10) {
    return epilogue;
  }
  // conv2d.with_epilogue passes residual, residual_scale, residual_op, alpha,
  // activation, activation_a and activation_b before the output
  VK_CHECK_COND(args.size() == 17, "Unexpected conv2d argument count");
  if (graph.val_is_not_none(args[9])) {
    epilogue.residual = args[9];
    epilogue.residual_op =
        residual_op_from_string(graph.extract_string(args[11]));
    epilogue.residual_alpha = graph.extract_scalar<float>(args[12]);
  }
  if (graph.val_is_not_none(args[10])) {
    epilogue.scale = args[10];
  }
  const std::string activation = graph.extract_string(args[13]);
  if (activation != "none") {
    epilogue.activation = activation_from_string(activation);
    epilogue.activation_a = graph.extract_scalar<float>(args[14]);
    epilogue.activation_b = graph.extract_scalar<float>(args[15]);
  }
  check_conv2d_epilogue(graph, epilogue, args[1], args.back());
  return epilogue;
}

void conv2d_with_epilogue_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef in = args[0];
  const ValueRef weight_data = args[1];
  const ValueRef bias = args[2];
  const ValueRef stride = args[3];
  const ValueRef padding = args[4];
  const ValueRef dilation = args[5];
  const ValueRef transposed = args[6];
  const ValueRef output_padding = args[7];
  const ValueRef groups = args[8];
  const ValueRef out = args.back();

  VK_CHECK_COND(graph.dim_of(in) == 4);
  const Conv2dEpilogue epilogue = conv2d_epilogue_from_args(graph, args);

  const bool transposed_val = graph.get_bool(transposed);
  const int64_t groups_val = graph.get_int(groups);

  const Conv2dMethod method =
      get_conv2d_method(graph, weight_data, groups_val, transposed_val);

  // Use tiled path for all pointwise conv2d
  if (method == Conv2dMethod::Pointwise) {
    return conv2d_pw_impl(
        graph,
        in,
        weight_data,
        bias,
        stride,
        padding,
        out,
        transposed_val,
        epilogue);
  }

  if (method == Conv2dMethod::Depthwise) {
    return conv2d_dw_impl(
        graph, in, weight_data, bias, stride, padding, dilation, out, epilogue);
  }

  const Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      weight_data,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);

  const bool fast_tiled = can_use_fast_tiled_conv2d(
      graph, in, out, kernel_params, transposed_val, groups_val);
  // SlidingWindow conv2d: route to the im2col + GEMM path when the heuristic
  // indicates it is beneficial, falling back to the direct convolution shader
  // otherwise.
  const bool use_im2col =
      method == Conv2dMethod::SlidingWindow &&
      should_use_conv2d_im2col(
          graph, weight_data, groups_val, kernel_params, out, fast_tiled);
  if (use_im2col) {
    return conv2d_gemm_impl(
        graph, in, weight_data, bias, stride, padding, dilation, out, epilogue);
  }

  return conv2d_direct_impl(
      graph,
      in,
      weight_data,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out,
      epilogue);
}

} // namespace vkcompute
