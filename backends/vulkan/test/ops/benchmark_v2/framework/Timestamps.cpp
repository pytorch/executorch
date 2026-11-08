// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Timestamps.h>

#include <vector>

namespace vkcompute::shader_profiling {

Timestamps::Timestamps(api::Context& context)
    : device_(context.device()),
      period_ns_(context.adapter_ptr()->timestamp_period()) {
  uint32_t count = 0;
  const VkPhysicalDevice physical = context.adapter_ptr()->physical_handle();
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
  valid_bits_ = families.at(context.queue().family_index).timestampValidBits;
  if (valid_bits_ > 0 && period_ns_ > 0) {
    const VkQueryPoolCreateInfo info{
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,
        VK_QUERY_TYPE_TIMESTAMP,
        2,
        0};
    VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &pool_));
  }
}

Timestamps::~Timestamps() {
  if (pool_ != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device_, pool_, nullptr);
  }
}

bool Timestamps::available() const {
  return pool_ != VK_NULL_HANDLE;
}

double Timestamps::period_ns() const {
  return period_ns_;
}

uint32_t Timestamps::valid_bits() const {
  return valid_bits_;
}

void Timestamps::begin(const vkapi::CommandBuffer& cmd) const {
  cmd.reset_querypool(pool_, 0, 2);
  cmd.write_timestamp(pool_, 0);
}

void Timestamps::end(const vkapi::CommandBuffer& cmd) const {
  cmd.write_timestamp(pool_, 1);
}

double Timestamps::elapsed_us() const {
  uint64_t ticks[2];
  VK_CHECK(vkGetQueryPoolResults(
      device_,
      pool_,
      0,
      2,
      sizeof(ticks),
      ticks,
      sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  return elapsed_us(ticks[0], ticks[1]);
}

double Timestamps::elapsed_us(uint64_t start_ticks, uint64_t end_ticks) const {
  const uint64_t mask =
      valid_bits_ == 64 ? UINT64_MAX : (uint64_t(1) << valid_bits_) - 1;
  return static_cast<double>((end_ticks - start_ticks) & mask) * period_ns_ /
      1000.0;
}

} // namespace vkcompute::shader_profiling
