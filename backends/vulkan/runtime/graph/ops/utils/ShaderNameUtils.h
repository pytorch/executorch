/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <string>

namespace vkcompute {

constexpr size_t kShaderNameReserve = 64u;

void add_storage_type_suffix(
    std::string& kernel_name,
    const utils::StorageType storage_type);

void add_dtype_suffix(std::string& kernel_name, const vkapi::ScalarType dtype);

// Selects the per-token zero-point shader binding variant by the dtype the
// zero-point tensor was allocated with: "_zpint8" when the tensor is int8
// (rgba8i integer image), "_zpinherit" when it follows the inference float
// dtype (rgba32f/rgba16f, matching the scale). Matches the ZP_DTYPE_MODE
// codegen axis used by the dq8ca qparams shaders.
void add_zp_dtype_mode_suffix(
    std::string& kernel_name,
    const vkapi::ScalarType zp_dtype);

void add_ndim_suffix(std::string& kernel_name, const size_t ndim);

void add_packed_dim_suffix(std::string& kernel_name, const int32_t packed_dim);

} // namespace vkcompute
