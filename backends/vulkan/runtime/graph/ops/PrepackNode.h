/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace at {
namespace native {
namespace vulkan {

class ComputeGraph;

/*
 * Represents a single prepacking op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of shaders transferring necessary data (such as weights and biases)
 * to the GPU.
 */
class PrepackNode {
  friend class ComputeGraph;

 public:
  PrepackNode(ValueRef tref, ValueRef packed) : tref_{tref}, packed_{packed} {}

  virtual ~PrepackNode() = default;

 protected:
  ValueRef tref_;
  ValueRef packed_;

 public:
  virtual void encode(ComputeGraph* graph) const = 0;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
