/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/util/read_file.h>

#include <executorch/runtime/platform/log.h>

#include <stdio.h>
#include <memory>

namespace torch {
namespace executor {
namespace util {

__ET_NODISCARD Error read_file_content(
    const char* file_name,
    std::shared_ptr<char>* file_data,
    size_t* file_length) {
  FILE* file;
  unsigned long fileLen;

  // Open file
  file = fopen(file_name, "rb");
  if (!file) {
    ET_LOG(Error, "Unable to open file %s\n", file_name);
    return Error::NotSupported;
  }

  // Get file length
  fseek(file, 0, SEEK_END);
  fileLen = ftell(file);
  fseek(file, 0, SEEK_SET);

  // Allocate memory
  auto ptr = std::shared_ptr<char>(
      new char[fileLen + 1], std::default_delete<char[]>());
  if (!ptr) {
    ET_LOG(Error, "Unable to allocate memory to read file %s\n", file_name);
    fclose(file);
    return Error::NotSupported;
  }

  // Read file contents into buffer
  fread(ptr.get(), fileLen, 1, file);
  fclose(file);

  *file_data = ptr;
  *file_length = fileLen;
  return Error::Ok;
}

__ET_DEPRECATED std::shared_ptr<char> read_file_content(const char* name) {
  std::shared_ptr<char> file_data;
  size_t file_length;
  Error status = read_file_content(name, &file_data, &file_length);
  if (status == Error::Ok) {
    return file_data;
  } else {
    return nullptr;
  }
}

} // namespace util
} // namespace executor
} // namespace torch
