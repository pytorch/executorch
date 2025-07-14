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
 * A DataLoader that loads segments from a file descriptor, allocating the
 * memory with `malloc()`. This data loader is used when ET is running in a
 * process that does not have access to the filesystem, and the caller is able
 * to open the file and pass the file descriptor.
 *
 * Note that this will keep the file open for the duration of its lifetime, to
 * avoid the overhead of opening it again for every load() call.
 */
class FileDescriptorDataLoader final : public executorch::runtime::DataLoader {
 public:
  /**
   * Creates a new FileDescriptorDataLoader that wraps the named file
   * descriptor, and the ownership of the file descriptor is passed.
   *
   * @param[in] file_descriptor_uri File descriptor with the prefix "fd:///",
   *     followed by the file descriptor number.
   * @param[in] alignment Alignment in bytes of pointers returned by this
   *     instance. Must be a power of two.
   *
   * @returns A new FileDescriptorDataLoader on success.
   * @retval Error::InvalidArgument `alignment` is not a power of two.
   * @retval Error::AccessFailed `file_descriptor_uri` is incorrectly formatted,
   * or its size could not be found.
   * @retval Error::MemoryAllocationFailed Internal memory allocation failure.
   */
  static executorch::runtime::Result<FileDescriptorDataLoader>
  fromFileDescriptorUri(
      const char* file_descriptor_uri,
      size_t alignment = alignof(std::max_align_t));

  // Movable to be compatible with Result.
  FileDescriptorDataLoader(FileDescriptorDataLoader&& rhs) noexcept
      : file_descriptor_uri_(rhs.file_descriptor_uri_),
        file_size_(rhs.file_size_),
        alignment_(rhs.alignment_),
        fd_(rhs.fd_) {
    const_cast<const char*&>(rhs.file_descriptor_uri_) = nullptr;
    const_cast<size_t&>(rhs.file_size_) = 0;
    const_cast<std::align_val_t&>(rhs.alignment_) = {};
    const_cast<int&>(rhs.fd_) = -1;
  }

  ~FileDescriptorDataLoader() override;

  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo& segment_info) const override;

  ET_NODISCARD executorch::runtime::Result<size_t> size() const override;

  ET_NODISCARD executorch::runtime::Error load_into(
      size_t offset,
      size_t size,
      ET_UNUSED const SegmentInfo& segment_info,
      void* buffer) const override;

 private:
  FileDescriptorDataLoader(
      int fd,
      size_t file_size,
      size_t alignment,
      const char* file_descriptor_uri)
      : file_descriptor_uri_(file_descriptor_uri),
        file_size_(file_size),
        alignment_{alignment},
        fd_(fd) {}

  // Not safely copyable.
  FileDescriptorDataLoader(const FileDescriptorDataLoader&) = delete;
  FileDescriptorDataLoader& operator=(const FileDescriptorDataLoader&) = delete;
  FileDescriptorDataLoader& operator=(FileDescriptorDataLoader&&) = delete;

  const char* const file_descriptor_uri_; // Owned by the instance.
  const size_t file_size_;
  const std::align_val_t alignment_;
  const int fd_; // Owned by the instance.
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::FileDescriptorDataLoader;
} // namespace util
} // namespace executor
} // namespace torch
