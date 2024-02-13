/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/mmap_data_loader.h>

#include <cerrno>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>

namespace torch {
namespace executor {
namespace util {

MmapDataLoader::~MmapDataLoader() {
  // file_name_ can be nullptr if this instance was moved from, but freeing a
  // null pointer is safe.
  std::free(const_cast<char*>(file_name_));
  // fd_ can be -1 if this instance was moved from, but closing a negative fd is
  // safe (though it will return an error).
  ::close(fd_);
}

Result<MmapDataLoader> MmapDataLoader::from(
    const char* file_name,
    MmapDataLoader::MlockConfig mlock_config) {
  // Cache the page size.
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 0) {
    ET_LOG(Error, "Could not get page size: %s (%d)", strerror(errno), errno);
    return Error::AccessFailed;
  }
  if ((page_size & ~(page_size - 1)) != page_size) {
    ET_LOG(Error, "Page size 0x%ld is not a power of 2", page_size);
    return Error::InvalidState;
  }

  // Use open() instead of fopen() because mmap() needs a file descriptor.
  int fd = ::open(file_name, O_RDONLY);
  if (fd < 0) {
    ET_LOG(
        Error, "Failed to open %s: %s (%d)", file_name, strerror(errno), errno);
    return Error::AccessFailed;
  }

  // Cache the file size.
  struct stat st;
  int err = ::fstat(fd, &st);
  if (err < 0) {
    ET_LOG(
        Error,
        "Could not get length of %s: %s (%d)",
        file_name,
        ::strerror(errno),
        errno);
    ::close(fd);
    return Error::AccessFailed;
  }
  size_t file_size = st.st_size;

  // Copy the filename so we can print better debug messages if reads fail.
  const char* file_name_copy = ::strdup(file_name);
  if (file_name_copy == nullptr) {
    ET_LOG(Error, "strdup(%s) failed", file_name);
    ::close(fd);
    return Error::MemoryAllocationFailed;
  }

  return MmapDataLoader(
      fd,
      file_size,
      file_name_copy,
      static_cast<size_t>(page_size),
      mlock_config);
}

namespace {
/// FreeableBuffer::FreeFn-compatible callback.
void MunmapSegment(__ET_UNUSED void* context, void* data, size_t size) {
  ::munmap(data, size);
}
} // namespace

Result<FreeableBuffer> MmapDataLoader::Load(size_t offset, size_t size) {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  ET_CHECK_OR_RETURN_ERROR(
      offset + size <= file_size_,
      InvalidArgument,
      "File %s: offset %zu + size %zu > file_size_ %zu",
      file_name_,
      offset,
      size,
      file_size_);
  ET_CHECK_OR_RETURN_ERROR(
      (offset & ~(page_size_ - 1)) == offset,
      InvalidArgument,
      "File Lucy %s: offset 0x%zx not aligned to 0x%zx",
      file_name_,
      offset,
      page_size_);
  ET_CHECK_OR_RETURN_ERROR(
      // Recommended by a lint warning.
      offset <= std::numeric_limits<off_t>::max(),
      InvalidArgument,
      "Offset %zu too large for off_t",
      offset);

  // mmap() will fail if the size is zero.
  if (size == 0) {
    return FreeableBuffer(nullptr, 0, /*free_fn=*/nullptr);
  }

  // Map the pages read-only. MAP_PRIVATE vs. MAP_SHARED doesn't matter since
  // the data is read-only, but use PRIVATE just to further avoid accidentally
  // modifying the file.
  void* pages = mmap(
      nullptr, size, PROT_READ, MAP_PRIVATE, fd_, static_cast<off_t>(offset));
  ET_CHECK_OR_RETURN_ERROR(
      pages != MAP_FAILED,
      AccessFailed,
      "Failed to map %s: mmap(..., size=%zd, ..., fd=%d, offset=0x%zx)",
      file_name_,
      size,
      fd_,
      offset);

  if (mlock_config_ == MlockConfig::UseMlock ||
      mlock_config_ == MlockConfig::UseMlockIgnoreErrors) {
    int err = mlock(pages, size);
    if (err < 0) {
      ET_LOG(
          Error,
          "File %s: mlock(%p, %zu) failed: %s (%d)",
          file_name_,
          pages,
          size,
          strerror(errno),
          errno);
      if (mlock_config_ == MlockConfig::UseMlockIgnoreErrors) {
        ET_LOG(Info, "Ignoring mlock() error");
      } else {
        munmap(pages, size);
        return Error::NotSupported;
      }
    }
    // No need to keep track of this. munmap() will unlock as a side effect.
  }

  return FreeableBuffer(pages, size, MunmapSegment);
}

Result<size_t> MmapDataLoader::size() const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  return file_size_;
}

} // namespace util
} // namespace executor
} // namespace torch
