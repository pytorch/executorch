/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/compiler.h>
#include <memory>
#include <string>

namespace torch {
namespace executor {
namespace util {

/**
 * Read the data from the file given name.
 *
 * The returned pointer pointing to the memory address containing the data, and
 * the file length is the length of data.
 *
 * @param[in] file_name The name of the file to be read.
 * @param[out] file_data The file data, if read successfully.
 * @param[out] file_length The length of file_data, in bytes, if read
 * successfully.
 *
 * @returns Error::Ok if the file is read successfully, file_data point to the
 * data and file_length is the correct length of file_data. Other values on
 * failure.
 */
__ET_NODISCARD Error read_file_content(
    const char* file_name,
    std::shared_ptr<char>* file_data,
    size_t* file_length);

/**
 * Read the data from the file given name.
 *
 * The returned pointer pointing to the memory address containing the data.
 *
 * This function is deprecated, and should use the above function instead to
 * read file content.
 *
 * @param[in] name The name of the file to be read.
 *
 * @returns The pointer to file data, if read successfully. Otherwise null_ptr.
 */
__ET_DEPRECATED std::shared_ptr<char> read_file_content(const char* name);

} // namespace util
} // namespace executor
} // namespace torch
