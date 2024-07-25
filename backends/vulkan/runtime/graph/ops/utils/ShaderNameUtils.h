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
void add_storage_type_suffix(
    std::string& kernel_name,
    const api::vTensor& tensor);

void add_dtype_suffix(std::string& kernel_name, const vkapi::ScalarType dtype);
void add_dtype_suffix(std::string& kernel_name, const api::vTensor& tensor);

void add_ndim_suffix(std::string& kernel_name, const size_t ndim);
void add_ndim_suffix(std::string& kernel_name, const api::vTensor& tensor);

void add_memory_layout_suffix(
    std::string& kernel_name,
    const utils::GPUMemoryLayout layout);
void add_memory_layout_suffix(
    std::string& kernel_name,
    const api::vTensor& tensor);

} // namespace vkcompute
