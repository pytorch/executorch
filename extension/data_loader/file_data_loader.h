/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {

/**
 * A DataLoader that loads segments from a file, allocating the memory
 * with `malloc()`.
 *
 * Note that this will keep the file open for the duration of its lifetime, to
 * avoid the overhead of opening it again for every load() call.
 */
class FileDataLoader : public executorch::runtime::DataLoader {
 public:
  /**
   * Creates a new FileDataLoader that wraps the named file.
   *
   * @param[in] file_name Path to the file to read from.
   * @param[in] alignment Alignment in bytes of pointers returned by this
   *     instance. Must be a power of two.
   *
   * @returns A new FileDataLoader on success.
   * @retval Error::InvalidArgument `alignment` is not a power of two.
   * @retval Error::AccessFailed `file_name` could not be opened, or its size
   *     could not be found.
   * @retval Error::MemoryAllocationFailed Internal memory allocation failure.
   */
  static executorch::runtime::Result<FileDataLoader> from(
      const char* file_name,
      size_t alignment = alignof(std::max_align_t));

  /// DEPRECATED: Use the lowercase `from()` instead.
  __ET_DEPRECATED static executorch::runtime::Result<FileDataLoader> From(
      const char* file_name,
      size_t alignment = alignof(std::max_align_t)) {
    return from(file_name, alignment);
  }

  // Movable to be compatible with Result.
  FileDataLoader(FileDataLoader&& rhs) noexcept
      : file_name_(rhs.file_name_),
        file_size_(rhs.file_size_),
        alignment_(rhs.alignment_),
        fd_(rhs.fd_) {
    rhs.file_name_ = nullptr;
    rhs.file_size_ = 0;
    rhs.alignment_ = 0;
    rhs.fd_ = -1;
  }

  ~FileDataLoader() override;

  __ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo& segment_info) override;

  __ET_NODISCARD executorch::runtime::Result<size_t> size() const override;

 private:
  FileDataLoader(
      int fd,
      size_t file_size,
      size_t alignment,
      const char* file_name)
      : file_name_(file_name),
        file_size_(file_size),
        alignment_(alignment),
        fd_(fd) {}

  // Not safely copyable.
  FileDataLoader(const FileDataLoader&) = delete;
  FileDataLoader& operator=(const FileDataLoader&) = delete;
  FileDataLoader& operator=(FileDataLoader&&) = delete;

  const char* file_name_; // Owned by the instance.
  size_t file_size_;
  size_t alignment_;
  int fd_; // Owned by the instance.
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::FileDataLoader;
} // namespace util
} // namespace executor
} // namespace torch
