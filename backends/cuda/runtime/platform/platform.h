
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

/**
 * Writes a shared library to a new file in the system temporary directory.
 *
 * The file gets a random name and is created exclusively, so an existing file
 * is never opened or overwritten. Outside Windows, only the current user can
 * read or write it. The directory is std::filesystem::temp_directory_path(),
 * which honors TMPDIR on POSIX. The caller removes the file when it is no
 * longer needed.
 *
 * @param[in] data The library contents.
 * @param[in] size The size of the library contents, in bytes.
 * @returns The path of the new file, or Error::AccessFailed if it could not be
 *     created or written.
 */
executorch::runtime::Result<std::filesystem::path> write_temp_library(
    const void* data,
    size_t size);

executorch::runtime::Result<void*> get_function(
    void* lib_handle,
    const std::string& fn_name);

int32_t get_process_id();

void* aligned_alloc(size_t alignment, size_t size);

void aligned_free(void* ptr);

} // namespace cuda
} // namespace backends
} // namespace executorch
