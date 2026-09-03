/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/runtime/core/portable_type/half.h>

namespace vkcompute {

std::vector<int64_t> calc_out_mean_sizes(
    const std::vector<int64_t>& self_sizes,
    int64_t normalized_shape_dim) {
  std::vector<int64_t> output_size = self_sizes;
  int64_t self_dim = self_sizes.size();
  for (int64_t i = 0; i < normalized_shape_dim; ++i) {
    output_size.at(self_dim - i - 1) = 1;
  }
  return output_size;
}

void resize_native_layer_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mean = args.at(0).refs.at(1);
  const ValueRef rstd = args.at(0).refs.at(2);
  const ValueRef in = args.at(1).refs.at(0);
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  const auto normalized_shape_dim =
      graph->get_int_list(extra_args.at(0))->size();

  const std::vector<int64_t> mean_size =
      calc_out_mean_sizes(in_sizes, normalized_shape_dim);

  graph->virtual_resize(out, in_sizes);
  graph->virtual_resize(mean, mean_size);
  graph->virtual_resize(rstd, mean_size);
}

utils::uvec3 layer_norm_buffer_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef mean_tensor = args.at(0).refs.at(1);
  const uint32_t num_rows =
      utils::safe_downcast<uint32_t>(graph->numel_of(mean_tensor));
  return {1u, num_rows, 1u};
}

utils::uvec3 layer_norm_buffer_local_wg_size(
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
  return {64u, 1u, 1u};
}

// Builds a TensorRef of `sizes` filled with `value`, for synthesizing a
// layer_norm affine parameter the caller did not supply. Ownership of the
// buffer passes to the FreeableBuffer, then to the TensorRef, then to the
// graph.
namespace etensor = executorch::runtime::etensor;

ValueRef constant_affine_tensorref(
    ComputeGraph& graph,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const float value) {
  int64_t numel = 1;
  for (int64_t d : sizes) {
    numel *= d;
  }
  const size_t total_bytes =
      static_cast<size_t>(numel) * vkapi::element_size(dtype);
  auto* data = new uint8_t[total_bytes]();
  if (value != 0.0f) {
    switch (dtype) {
      case vkapi::kFloat: {
        auto* typed = reinterpret_cast<float*>(data);
        std::fill(typed, typed + numel, value);
        break;
      }
      case vkapi::kHalf: {
        auto* typed = reinterpret_cast<etensor::Half*>(data);
        std::fill(typed, typed + numel, etensor::Half(value));
        break;
      }
      default:
        delete[] data;
        VK_THROW(
            "native_layer_norm cannot synthesize an affine parameter of dtype ",
            static_cast<int>(dtype));
    }
  }
  executorch::runtime::FreeableBuffer buffer(
      data, total_bytes, [](void* /*ctx*/, void* ptr, size_t /*size*/) {
        delete[] static_cast<uint8_t*>(ptr);
      });
  return graph.add_tensorref(sizes, dtype, std::move(buffer));
}

void add_native_layer_norm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef normalized_shape,
    const ValueRef weight_data,
    const ValueRef bias_data,
    const ValueRef eps,
    const ValueRef out) {
  const auto normalized_shape_dim =
      graph.get_int_list(normalized_shape)->size();
  if (normalized_shape_dim > 1) {
    VK_THROW("native_layer_norm only supports normalized_shape with dim == 1");
  }

  // The shader reads a weight and a bias binding unconditionally, but either
  // affine parameter can legitimately be absent:
  //   - nn.LayerNorm(bias=False) (e.g. HF Gemma4 audio_encoder) has no bias.
  //   - F.layer_norm called with neither, where the caller applies its own
  //     scale and shift afterwards (e.g. kokoro's AdaLayerNorm), has neither.
  // Synthesize whatever is missing: a unit weight and a zero bias reproduce
  // out = (x - mean) * rstd exactly.
  const std::vector<int64_t> affine_sizes = graph.val_is_none(weight_data)
      ? std::vector<int64_t>{*graph.get_int_list(normalized_shape)}
      : graph.sizes_of(weight_data);
  const vkapi::ScalarType affine_dtype = graph.val_is_none(weight_data)
      ? graph.dtype_of(in)
      : graph.dtype_of(weight_data);

  ValueRef synthesized_weight = weight_data;
  if (graph.val_is_none(weight_data)) {
    synthesized_weight =
        constant_affine_tensorref(graph, affine_sizes, affine_dtype, 1.0f);
  }
  ValueRef synthesized_bias = bias_data;
  if (graph.val_is_none(bias_data)) {
    synthesized_bias =
        constant_affine_tensorref(graph, affine_sizes, affine_dtype, 0.0f);
  }

  ValueRef arg_weight = prepack_standard_like(graph, synthesized_weight, in);
  ValueRef arg_bias = prepack_standard_like(graph, synthesized_bias, in);

  const auto out_val = graph.get_value_list(out);
  const ValueRef out_tensor = out_val->at(0);
  const ValueRef mean_tensor = out_val->at(1);
  const ValueRef rstd_tensor = out_val->at(2);

  float epsilon = graph.extract_scalar<float>(eps);

  std::string kernel_name("native_layer_norm");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out_tensor));
  add_dtype_suffix(kernel_name, graph.dtype_of(out_tensor));

  const bool is_buffer = graph.is_buffer_storage(in);

  if (!is_buffer) {
    VK_CHECK_COND(check_same_packed_dim(graph, in, out_tensor));
  }

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out_tensor), graph.meta_ubo(in)};
  vkapi::SpecVarList spec_constants = {graph.hashed_layout_of(in)};

  if (is_buffer) {
    param_ubos.append(graph.meta_ubo(mean_tensor));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      is_buffer ? layer_norm_buffer_global_wg_size
                : default_pick_global_wg_size,
      is_buffer ? layer_norm_buffer_local_wg_size : default_pick_local_wg_size,
      // Inputs and Outputs
      {{{out_tensor, mean_tensor, rstd_tensor}, vkapi::kWrite},
       {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {PushConstantDataInfo(&epsilon, sizeof(epsilon))},
      // Specialization Constants
      spec_constants,
      // Resize Args
      {normalized_shape},
      // Resizing Logic
      resize_native_layer_norm_node));
}

void native_layer_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_native_layer_norm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.native_layer_norm.default, native_layer_norm);
}

} // namespace vkcompute
