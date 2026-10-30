// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "device_info.h"
#include "cm_utils.h"

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <iostream>

namespace executorch {
namespace vulkan {
namespace prototyping {

void print_device_info() {
  auto* adapter = vkcompute::api::context()->adapter_ptr();

  std::cout << "=== Vulkan adapter ===\n";
  std::cout << "device_name              : " << adapter->device_name() << "\n";
  std::cout << "is_integrated_gpu        : "
            << (adapter->is_integrated_gpu() ? "yes" : "no") << "\n";
  std::cout << "subgroup_size            : " << adapter->subgroup_size()
            << "\n";
  std::cout << "min_subgroup_size        : " << adapter->min_subgroup_size()
            << "\n";
  std::cout << "max_subgroup_size        : " << adapter->max_subgroup_size()
            << "\n";
  std::cout << "supports_cooperative_mat : "
            << (adapter->supports_cooperative_matrix() ? "yes" : "no") << "\n";
  std::cout << "supports_int8_dot_product: "
            << (adapter->supports_int8_dot_product() ? "yes" : "no") << "\n";

  queryCooperativeMatrixProperties();
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
