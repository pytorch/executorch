/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace vkcompute {

class ComputeGraph;

constexpr uint32_t kMaxPushConstantSize = 128;
/*
 * Represents a push constant data entry
 * Which is either shared pointer to a tensor's uniform data with an attribute
 * Or data with a maximum size of 16 bytes
 */
class PushConstantDataInfo {
  std::shared_ptr<api::vTensor::UniformData> tensorUniformData;
  union Payload {
    struct {
      api::vTensor::Attribute attr;
    };
    struct {
      uint8_t data[16];
      uint32_t dataSize;
    };
  };

  Payload payload_;
  // The value in a compute graph that this push constant data is associated
  // with, if any.
  ValueRef value_ = kDummyValueRef;

 public:
  explicit PushConstantDataInfo(
      const std::shared_ptr<api::vTensor::UniformData>& tensorUniformData,
      api::vTensor::Attribute attr)
      : tensorUniformData(tensorUniformData) {
    payload_.attr = attr;
  }

  explicit PushConstantDataInfo(
      const void* data,
      uint32_t dataLen,
      uint32_t pushConstantLen = 0)
      : tensorUniformData(nullptr) {
    VK_CHECK_COND(
        dataLen <= 16, "Single push constant data size must be <= 16 bytes");
    payload_.dataSize = pushConstantLen ? pushConstantLen : dataLen;
    memcpy(payload_.data, data, dataLen);
  }

  /*
   * Function writes push constant data to the destination buffer
   */
  uint32_t write(
      void* dst,
      const uint32_t dst_offset,
      const uint32_t max_dst_size) const;

  inline bool is_tensor_metadata() const noexcept {
    return tensorUniformData != nullptr;
  }

  inline void set_value(ValueRef value) noexcept {
    value_ = value;
  }

  inline ValueRef value() const noexcept {
    return value_;
  }
};

} // namespace vkcompute
