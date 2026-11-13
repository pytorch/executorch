/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/wmma/WmmaLinear.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>

namespace vkcompute {

namespace {

#ifdef VK_KHR_cooperative_matrix
CoopmatComponentType to_component_type(const VkComponentTypeKHR t) {
  switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return CoopmatComponentType::kFp16;
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return CoopmatComponentType::kFp32;
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return CoopmatComponentType::kSint8;
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return CoopmatComponentType::kSint32;
    default:
      return CoopmatComponentType::kOther;
  }
}

void add_coopmat_shapes(
    CoopmatDeviceKey& key,
    const std::vector<VkCooperativeMatrixPropertiesKHR>& properties) {
  for (const VkCooperativeMatrixPropertiesKHR& p : properties) {
    if (p.scope != VK_SCOPE_SUBGROUP_KHR || p.AType != p.BType ||
        p.CType != p.ResultType) {
      continue;
    }
    key.coopmat_shapes.push_back(
        {p.MSize,
         p.NSize,
         p.KSize,
         to_component_type(p.AType),
         to_component_type(p.ResultType)});
  }
}
#endif /* VK_KHR_cooperative_matrix */

} // namespace

CoopmatDeviceKey coopmat_device_key(vkapi::Adapter* adapter) {
  CoopmatDeviceKey key;
  key.vendor_id = adapter->vendor_id();
  key.driver_id = adapter->driver_id();
  key.min_subgroup_size = adapter->min_subgroup_size();
  key.max_subgroup_size = adapter->max_subgroup_size();
  key.subgroup_size_control = adapter->supports_subgroup_size_control();
  key.compute_required_subgroup_size =
      adapter->supports_required_subgroup_size_for_compute();
  key.full_subgroups = adapter->compute_pipeline_cache().full_subgroups();
  key.vulkan_memory_model = adapter->supports_vulkan_memory_model();
  key.coopmat = adapter->supports_cooperative_matrix();
  key.int8_coopmat = adapter->supports_int8_cooperative_matrix();
  key.max_shared_bytes = adapter->max_compute_shared_memory_size();
#ifdef VK_KHR_cooperative_matrix
  add_coopmat_shapes(key, adapter->cooperative_matrix_properties());
#endif /* VK_KHR_cooperative_matrix */
  return key;
}

CoopmatDeviceKey coopmat_device_key(ComputeGraph* graph) {
  return coopmat_device_key(graph->context()->adapter_ptr());
}

WmmaStorage to_wmma_storage(const utils::StorageType storage) {
  switch (storage) {
    case utils::kBuffer:
      return WmmaStorage::kBuffer;
    case utils::kTexture2D:
      return WmmaStorage::kTexture2D;
    case utils::kTexture3D:
      return WmmaStorage::kTexture3D;
    default:
      return WmmaStorage::kOther;
  }
}

WmmaShape wmma_shape(
    ComputeGraph& graph,
    const WmmaOp op,
    const ValueRef fp_input,
    const ValueRef output,
    const ValueRef group_size,
    const ValueRef bias,
    const utils::StorageType weight_storage) {
  const std::vector<int64_t> out_sizes = graph.sizes_of(output);
  WmmaShape s;
  s.op = op;
  s.N = utils::val_at(-1, out_sizes);
  s.M = utils::val_at(-2, out_sizes);
  s.K = utils::val_at(-1, graph.sizes_of(fp_input));
  s.group_size = graph.extract_scalar<int64_t>(group_size);
  for (int64_t d = 0; d < graph.dim_of(output) - 2; d++) {
    s.batch *= utils::val_at(d, out_sizes);
  }
  s.gemv = is_gemv(&graph, fp_input);
  s.has_bias = !graph.val_is_none(bias);
  s.half = graph.dtype_of(output) == vkapi::kHalf;
  s.input = to_wmma_storage(graph.storage_type_of(fp_input));
  s.output = to_wmma_storage(graph.storage_type_of(output));
  s.weight = to_wmma_storage(weight_storage);
  s.io_width_packed = graph.packed_dim_of(output) == WHCN::kWidthDim &&
      graph.packed_dim_of(fp_input) == WHCN::kWidthDim;
  return s;
}

utils::StorageType predicted_q4_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> sizes = graph.sizes_of(weight_data);
  const int64_t K = utils::val_at(-1, sizes) * 2;
  const int64_t N = utils::val_at(-2, sizes);
  // Mirrors the 4-bit packed extents in prepack_quantized_linear_weight().
  const int64_t height = utils::div_up(N, int64_t(8));
  const int64_t width = utils::div_up(K, int64_t(4)) * 4;
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  return (width > int64_t(max_extent) * 4 || height > int64_t(max_extent))
      ? utils::kBuffer
      : utils::kTexture2D;
}

GlobalWorkGrid wmma_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const std::optional<WmmaTile> tile = wmma_tile_for_kernel(shader.kernel_name);
  if (!tile.has_value()) {
    return quantized_linear_gwg(graph, shader, args, resize_args);
  }
  const std::vector<int64_t> out_sizes = graph->sizes_of(args.at(0).refs.at(0));
  const uint32_t N =
      utils::safe_downcast<uint32_t>(utils::val_at(-1, out_sizes));
  const uint32_t M =
      utils::safe_downcast<uint32_t>(utils::val_at(-2, out_sizes));
  // The x extent is multiplied by the workgroup size because the framework
  // divides it by the local size.
  const uint32_t wg_size = tile->wg_size();
  return GlobalWorkGrid(
      {utils::div_up(N, tile->n) * wg_size, utils::div_up(M, tile->m), 1u},
      kTiledWorkGrid,
      LocalWorkGroup(wg_size, 1u, 1u));
}

} // namespace vkcompute
