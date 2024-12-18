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

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

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
};

/*
 * Represents a single shader execution op in a ML model.
 */
class DispatchNode final : public ExecuteNode {
  friend class ComputeGraph;

 public:
  explicit DispatchNode(
      ComputeGraph& graph,
      const vkapi::ShaderInfo& shader,
      const utils::uvec3& global_workgroup_size,
      const utils::uvec3& local_workgroup_size,
      const std::vector<ArgGroup>& args,
      const vkapi::ParamsBindList& params,
      const vkapi::SpecVarList& spec_vars = {},
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {},
      const std::vector<PushConstantDataInfo>& push_constants = {});

  ~DispatchNode() override = default;

  void encode(ComputeGraph* graph) override;

 protected:
  const vkapi::ShaderInfo shader_;
  const utils::uvec3 global_workgroup_size_;
  const utils::uvec3 local_workgroup_size_;
  const vkapi::ParamsBindList params_;
  const vkapi::SpecVarList spec_vars_;
  const std::vector<PushConstantDataInfo> push_constants_;

 public:
  operator bool() const {
    return shader_;
  }
};

} // namespace vkcompute
