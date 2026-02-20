
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <filesystem>
#include <string>

namespace executorch {
namespace backends {
namespace cuda {

executorch::runtime::Result<void*> load_library(
    const std::filesystem::path& path);

executorch::runtime::Error close_library(void* lib_handle);

executorch::runtime::Result<void*> get_function(
    void* lib_handle,
    const std::string& fn_name);

int32_t get_process_id();

void* aligned_alloc(size_t alignment, size_t size);

void aligned_free(void* ptr);

} // namespace cuda
} // namespace backends
} // namespace executorch
