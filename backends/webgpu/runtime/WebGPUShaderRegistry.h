/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace executorch::backends::webgpu {

struct WebGPUShaderInfo {
  std::string_view name;
  const char* source;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint32_t workgroup_size_z;
};

const WebGPUShaderInfo& get_webgpu_shader_info(std::string_view name);

} // namespace executorch::backends::webgpu
