/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_descriptor_data_loader.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace executorch {
namespace extension {

namespace {

static constexpr char kFdFilesystemPrefix[] = "fd:///";

/**
 * Returns true if the value is an integer power of 2.
 */
static bool is_power_of_2(size_t value) {
  return value > 0 && (value & ~(value - 1)) == value;
}

} // namespace

FileDescriptorDataLoader::~FileDescriptorDataLoader() {
  // file_descriptor_uri_ can be nullptr if this instance was moved from, but
  // freeing a null pointer is safe.
  std::free(const_cast<char*>(file_descriptor_uri_));
  // fd_ can be -1 if this instance was moved from, but closing a negative fd is
  // safe (though it will return an error).
  ::close(fd_);
}

static Result<int> getFDFromUri(const char* file_descriptor_uri) {
  // check if the uri starts with the prefix "fd://"
  ET_CHECK_OR_RETURN_ERROR(
      strncmp(
          file_descriptor_uri,
          kFdFilesystemPrefix,
          strlen(kFdFilesystemPrefix)) == 0,
      InvalidArgument,
      "File descriptor uri (%s) does not start with %s",
      file_descriptor_uri,
      kFdFilesystemPrefix);

  // strip "fd:///" from the uri
  int fd_len = strlen(file_descriptor_uri) - strlen(kFdFilesystemPrefix);
  char fd_without_prefix[fd_len + 1];
  memcpy(
      fd_without_prefix,
      &file_descriptor_uri[strlen(kFdFilesystemPrefix)],
      fd_len);
  fd_without_prefix[fd_len] = '\0';

  // check if remaining fd string is a valid integer
  int fd = ::atoi(fd_without_prefix);
  return fd;
}

Result<FileDescriptorDataLoader>
FileDescriptorDataLoader::fromFileDescriptorUri(
    const char* file_descriptor_uri,
    size_t alignment) {
  ET_CHECK_OR_RETURN_ERROR(
      is_power_of_2(alignment),
      InvalidArgument,
      "Alignment %zu is not a power of 2",
      alignment);

  auto parsed_fd = getFDFromUri(file_descriptor_uri);
  if (!parsed_fd.ok()) {
    return parsed_fd.error();
  }

  int fd = parsed_fd.get();

  // Cache the file size.
  struct stat st;
  int err = ::fstat(fd, &st);
  if (err < 0) {
    ET_LOG(
        Error,
        "Could not get length of %s: %s (%d)",
        file_descriptor_uri,
        ::strerror(errno),
        errno);
    ::close(fd);
    return Error::AccessFailed;
  }
  size_t file_size = st.st_size;

  // Copy the filename so we can print better debug messages if reads fail.
  const char* file_descriptor_uri_copy = ::strdup(file_descriptor_uri);
  if (file_descriptor_uri_copy == nullptr) {
    ET_LOG(Error, "strdup(%s) failed", file_descriptor_uri);
    ::close(fd);
    return Error::MemoryAllocationFailed;
  }

  return FileDescriptorDataLoader(
      fd, file_size, alignment, file_descriptor_uri_copy);
}

namespace {

inline void* et_aligned_alloc(size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}

inline void et_aligned_free(void* ptr, std::align_val_t alignment) {
  return ::operator delete(ptr, alignment);
}

/**
 * FreeableBuffer::FreeFn-compatible callback.
 *
 * `data` is the original buffer pointer.
 * `context` is the original alignment.
 *
 * `size` is unused.
 */
void FreeSegment(void* context, void* data, ET_UNUSED size_t size) {
  et_aligned_free(
      data,
      static_cast<std::align_val_t>(reinterpret_cast<uintptr_t>(context)));
}

} // namespace

Result<FreeableBuffer> FileDescriptorDataLoader::load(
    size_t offset,
    size_t size,
    ET_UNUSED const DataLoader::SegmentInfo& segment_info) const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  ET_CHECK_OR_RETURN_ERROR(
      offset + size <= file_size_,
      InvalidArgument,
      "File %s: offset %zu + size %zu > file_size_ %zu",
      file_descriptor_uri_,
      offset,
      size,
      file_size_);

  // Don't bother allocating/freeing for empty segments.
  if (size == 0) {
    return FreeableBuffer(nullptr, 0, /*free_fn=*/nullptr);
  }

  // Allocate memory for the FreeableBuffer.
  void* aligned_buffer = et_aligned_alloc(size, alignment_);
  if (aligned_buffer == nullptr) {
    ET_LOG(
        Error,
        "Reading from %s at offset %zu: et_aligned_alloc(%zu, %zu) failed",
        file_descriptor_uri_,
        offset,
        size,
        static_cast<size_t>(alignment_));
    return Error::MemoryAllocationFailed;
  }

  auto err = load_into(offset, size, segment_info, aligned_buffer);
  if (err != Error::Ok) {
    et_aligned_free(aligned_buffer, alignment_);
    return err;
  }

  // Pass the alignment as context to FreeSegment.
  return FreeableBuffer(
      aligned_buffer,
      size,
      FreeSegment,
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(static_cast<uintptr_t>(alignment_)));
}

Result<size_t> FileDescriptorDataLoader::size() const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  return file_size_;
}

ET_NODISCARD Error FileDescriptorDataLoader::load_into(
    size_t offset,
    size_t size,
    ET_UNUSED const SegmentInfo& segment_info,
    void* buffer) const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  ET_CHECK_OR_RETURN_ERROR(
      offset + size <= file_size_,
      InvalidArgument,
      "File %s: offset %zu + size %zu > file_size_ %zu",
      file_descriptor_uri_,
      offset,
      size,
      file_size_);
  ET_CHECK_OR_RETURN_ERROR(
      buffer != nullptr, InvalidArgument, "Provided buffer cannot be null");

  // Read the data into the aligned address.
  size_t needed = size;
  uint8_t* buf = reinterpret_cast<uint8_t*>(buffer);

  while (needed > 0) {
    // Reads on macOS will fail with EINVAL if size > INT32_MAX.
    const auto chunk_size = std::min<size_t>(
        needed, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const auto nread = ::pread(fd_, buf, chunk_size, offset);
    if (nread < 0 && errno == EINTR) {
      // Interrupted by a signal; zero bytes read.
      continue;
    }
    if (nread <= 0) {
      // nread == 0 means EOF, which we shouldn't see if we were able to read
      // the full amount. nread < 0 means an error occurred.
      ET_LOG(
          Error,
          "Reading from %s: failed to read %zu bytes at offset %zu: %s",
          file_descriptor_uri_,
          size,
          offset,
          nread == 0 ? "EOF" : strerror(errno));
      return Error::AccessFailed;
    }
    needed -= nread;
    buf += nread;
    offset += nread;
  }
  return Error::Ok;
}

} // namespace extension
} // namespace executorch
