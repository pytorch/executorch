/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/Utils.h>

#include <ATen/native/vulkan/impl/Common.h>

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
