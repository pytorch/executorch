// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// One-shot capability probe: prints the cooperative matrix configurations
// exposed by the active Vulkan device, plus the relevant adapter properties
// (subgroup size, iGPU vs dGPU, device name). Used to gate the coopmat
// implementations for quantized linear shaders.

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include "cm_utils.h"

#include <iostream>

using namespace executorch::vulkan::prototyping;

int main() {
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

  return 0;
}
