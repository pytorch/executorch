/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>

#include <functional>
#include <memory>

namespace vkcompute {
namespace vkapi {

//
// A Vulkan Runtime initializes a Vulkan instance and decouples the concept of
// Vulkan instance initialization from initialization of, and subsequent
// interactions with,  Vulkan [physical and logical] devices as a precursor to
// multi-GPU support.  The Vulkan Runtime can be queried for available Adapters
// (i.e. physical devices) in the system which in turn can be used for creation
// of a Vulkan Context (i.e. logical devices).  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
//

enum AdapterSelector {
  First,
};

struct RuntimeConfig final {
  bool enable_validation_messages;
  bool init_default_device;
  AdapterSelector default_selector;
  uint32_t num_requested_queues;
  std::string cache_data_path;
};

class Runtime final {
 public:
  explicit Runtime(const RuntimeConfig);

  // Do not allow copying. There should be only one global instance of this
  // class.
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  ~Runtime();

  using DeviceMapping = std::pair<PhysicalDevice, int32_t>;
  using AdapterPtr = std::unique_ptr<Adapter>;

 private:
  RuntimeConfig config_;

  VkInstance instance_;

  std::vector<DeviceMapping> device_mappings_;
  std::vector<AdapterPtr> adapters_;
  uint32_t default_adapter_i_;

  VkDebugReportCallbackEXT debug_report_callback_;

 public:
  inline VkInstance instance() const {
    return instance_;
  }

  inline Adapter* get_adapter_p() {
    VK_CHECK_COND(
        default_adapter_i_ >= 0 && default_adapter_i_ < adapters_.size(),
        "Pytorch Vulkan Runtime: Default device adapter is not set correctly!");
    return adapters_[default_adapter_i_].get();
  }

  inline Adapter* get_adapter_p(uint32_t i) {
    VK_CHECK_COND(
        i >= 0 && i < adapters_.size(),
        "Pytorch Vulkan Runtime: Adapter at index ",
        i,
        " is not available!");
    return adapters_[i].get();
  }

  inline uint32_t default_adapter_i() const {
    return default_adapter_i_;
  }

  using Selector =
      std::function<uint32_t(const std::vector<Runtime::DeviceMapping>&)>;
  uint32_t create_adapter(const Selector&);
};

std::string& set_and_get_pipeline_cache_data_path(const std::string& file_path);

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
Runtime* runtime();

// Used to share instance + devices between client code and ETVK
Adapter* set_and_get_external_adapter(
    const VkInstance instance = VK_NULL_HANDLE,
    const VkPhysicalDevice physical_device = VK_NULL_HANDLE,
    const VkDevice logical_device = VK_NULL_HANDLE);

} // namespace vkapi
} // namespace vkcompute
