// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <cstdint>

namespace vkcompute::shader_profiling {

class Timestamps final {
  VkDevice device_;
  VkQueryPool pool_ = VK_NULL_HANDLE;
  double period_ns_;
  uint32_t valid_bits_;

 public:
  explicit Timestamps(api::Context& context);
  Timestamps(const Timestamps&) = delete;
  ~Timestamps();

  Timestamps& operator=(const Timestamps&) = delete;

  bool available() const;
  double period_ns() const;
  uint32_t valid_bits() const;
  void begin(const vkapi::CommandBuffer& cmd) const;
  void end(const vkapi::CommandBuffer& cmd) const;
  double elapsed_us() const;
  double elapsed_us(uint64_t start_ticks, uint64_t end_ticks) const;
};

} // namespace vkcompute::shader_profiling
