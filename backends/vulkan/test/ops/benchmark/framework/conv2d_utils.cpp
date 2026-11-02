// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "conv2d_utils.h"

namespace executorch {
namespace vulkan {
namespace prototyping {

std::string make_test_case_conv_params_suffix(const Conv2dConfig& config) {
  std::string suffix;
  // Only print groups if not equal to 1
  if (config.groups != 1) {
    suffix += "g=" + std::to_string(config.groups);
    suffix += "  ";
  }

  suffix += "k=";
  if (config.kernel.h == config.kernel.w) {
    suffix += std::to_string(config.kernel.w);
  } else {
    suffix +=
        std::to_string(config.kernel.w) + "," + std::to_string(config.kernel.h);
  }
  // Only print stride if either dimension is not 1
  if (config.stride.h > 1 || config.stride.w > 1) {
    suffix += ",s=";
    if (config.stride.h == config.stride.w) {
      suffix += std::to_string(config.stride.w);
    } else {
      suffix += std::to_string(config.stride.w) + "," +
          std::to_string(config.stride.h);
    }
  }
  // Only print padding if either dimension is not 1
  if (config.padding.h != 1 || config.padding.w != 1) {
    suffix += ",p=";
    if (config.padding.h == config.padding.w) {
      suffix += std::to_string(config.padding.w);
    } else {
      suffix += std::to_string(config.padding.w) + "," +
          std::to_string(config.padding.h);
    }
  }
  // Only print dilation if either dimension is not 1
  if (config.dilation.h != 1 || config.dilation.w != 1) {
    suffix += ",d=";
    if (config.dilation.h == config.dilation.w) {
      suffix += std::to_string(config.dilation.w);
    } else {
      suffix += std::to_string(config.dilation.w) + "," +
          std::to_string(config.dilation.h);
    }
  }
  return suffix;
}

std::string to_string(const vkcompute::utils::StorageType storage_type) {
  switch (storage_type) {
    case vkcompute::utils::kTexture3D:
      return "Tex";
    case vkcompute::utils::kTexture2D:
      return "Tex2D";
    case vkcompute::utils::kBuffer:
      return "Buf";
  }
}

std::string make_test_case_name(
    const Conv2dConfig& config,
    const bool is_performance,
    const vkcompute::utils::StorageType fp_storage_type,
    const vkcompute::utils::StorageType int8_storage_type) {
  std::string test_case_name = is_performance ? "PERF  " : "ACCU  ";
  test_case_name += std::to_string(config.channels.in) + "->" +
      std::to_string(config.channels.out) +
      "  I=" + std::to_string(config.input_size.h) + "," +
      std::to_string(config.input_size.w) + "  " +
      make_test_case_conv_params_suffix(config);

  test_case_name +=
      "  " + to_string(fp_storage_type) + "->" + to_string(int8_storage_type);

  return test_case_name;
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
