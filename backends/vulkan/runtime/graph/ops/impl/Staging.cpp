/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/impl/Packing.h>

namespace at {
namespace native {
namespace vulkan {

void memcpy_to_mapping(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                    \
  case api::ScalarType::name:                                \
    memcpy_to_mapping_impl<ctype>(src, dst_mapping, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
}

void memcpy_from_mapping(
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                      \
  case api::ScalarType::name:                                  \
    memcpy_from_mapping_impl<ctype>(src_mapping, dst, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
}

void copy_ptr_to_staging(
    const void* src,
    api::StorageBuffer& staging,
    const size_t nbytes) {
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
  mapping.invalidate();
  memcpy_to_mapping(src, mapping, nbytes, staging.dtype());
}

void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes) {
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
  mapping.invalidate();
  memcpy_from_mapping(mapping, dst, nbytes, staging.dtype());
}

void encode_copy_to_vtensor(
    api::Context* context,
    api::StorageBuffer& staging,
    vTensor& tensor) {
  api::ShaderInfo shader = get_nchw_to_image_shader(tensor);
  api::PipelineBarrier pipeline_barrier{};
  packing::record_nchw_to_image_op(
      context,
      shader,
      staging.buffer(),
      tensor,
      pipeline_barrier,
      VK_NULL_HANDLE);
}

struct StagingParams final {
  api::utils::ivec3 extents;
  int32_t plane_size;
  api::utils::ivec2 channel_info;
};

StagingParams create_staging_params(const vTensor& t) {
  int32_t height = api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(t));
  int32_t width = api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(t));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(t));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  return {
      api::utils::make_ivec3(t.extents()),
      plane_size,
      {c_depth, channels},
  };
}

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor) {
  vTensor& t_out = graph.get_val(out_tensor).toTensor();
  VK_CHECK_COND(graph.get_val(in_staging).isStaging());

  api::ShaderInfo shader = get_nchw_to_image_shader(t_out);

  api::utils::uvec3 global_size = t_out.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::UniformParamsBuffer params(
      graph.context(), create_staging_params(t_out));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      shader,
      global_size,
      local_size,
      {out_tensor},
      {in_staging},
      std::move(params)));
}

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging) {
  vTensor& t_in = graph.get_val(in_tensor).toTensor();
  VK_CHECK_COND(graph.get_val(out_staging).isStaging());

  api::ShaderInfo shader = get_image_to_nchw_shader(t_in);

  api::utils::uvec3 global_size = t_in.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  StagingParams sp = create_staging_params(t_in);
  api::UniformParamsBuffer params(graph.context(), sp);

  // TODO(T181194784): These are workgroup sizes for special cases. Refactor the
  // calculation of workgroup sizes to a standalone function. We should use
  // scalar type to get the shader name, and use the shader name to get the
  // workgroup size.
  if (t_in.dtype() == api::ScalarType::QUInt8 ||
      t_in.dtype() == api::ScalarType::QInt8 || t_in.dtype() == api::kBool) {
    if (sp.plane_size % 4 == 0) {
      global_size.data[0u] = sp.plane_size / 4;
      global_size.data[1u] = 1;
      local_size.data[0u] *= local_size.data[1u];
      local_size.data[1u] = 1;
    } else {
      uint32_t numel = t_in.numel();
      global_size = {api::utils::div_up(numel, uint32_t(4)), 1u, 1u};
      local_size = {64u, 1u, 1u};
    }
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      shader,
      global_size,
      local_size,
      {in_tensor},
      {out_staging},
      std::move(params)));
}

api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst) {
  if (v_dst.is_quantized()) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        switch (v_dst.dtype()) {
          case api::ScalarType::QUInt8:
            return VK_KERNEL(nchw_to_image_uint8);
          case api::ScalarType::QInt8:
            return VK_KERNEL(nchw_to_image_int8);
          case api::ScalarType::QInt32:
            return VK_KERNEL(nchw_to_image_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_dst.dtype());
        }
      case api::StorageType::TEXTURE_2D:
        switch (v_dst.dtype()) {
          case api::ScalarType::QUInt8:
            return VK_KERNEL(nchw_to_image2d_uint8);
          case api::ScalarType::QInt8:
            return VK_KERNEL(nchw_to_image2d_int8);
          case api::ScalarType::QInt32:
            return VK_KERNEL(nchw_to_image2d_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_dst.dtype());
        }
      default:
        VK_THROW("No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        VK_THROW("Requested storage type must be a texture type.");
    }
  }

  if (v_dst.dtype() == api::kFloat) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(nchw_to_image2d);
      default:
        VK_THROW("No kernel available!");
    }
  } else if (v_dst.dtype() == api::kBool) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image_bool);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype!");
  }
}

api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src) {
  if (v_src.is_quantized() || v_src.dtype() == api::kBool) {
    auto plane_size =
        dim_at<Dim4D::Height>(v_src) * dim_at<Dim4D::Width>(v_src);
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        switch (v_src.dtype()) {
          case api::ScalarType::QUInt8:
          case api::ScalarType::QInt8:
          case api::kBool:
            return plane_size % 4 == 0 ? VK_KERNEL(image_to_nchw_quantized_mul4)
                                       : VK_KERNEL(image_to_nchw_uint);
          case api::ScalarType::QInt32:
            return VK_KERNEL(image_to_nchw_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_src.dtype());
        }
      default:
        VK_THROW("No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        VK_THROW("Requested storage type must be a texture type.");
    }
  }

  if (v_src.dtype() == api::kFloat) {
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(image2d_to_nchw);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype!");
  }
}

} // namespace vulkan
} // namespace native
} // namespace at
