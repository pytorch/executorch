/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>

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

// Some platforms (e.g. Xtensa) do not support pread() that we use to read the
// file at different offsets simultaneously from multiple threads not affecting
// each other. We list them below and use a workaround for them.
#if defined(__xtensa__)
#define ET_HAVE_PREAD 0
#endif // defined(__xtensa__)

#ifndef ET_HAVE_PREAD
#define ET_HAVE_PREAD 1
#endif // !ET_HAVE_PREAD

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace executorch {
namespace extension {

namespace {

/**
 * Returns true if the value is an integer power of 2.
 */
static bool is_power_of_2(size_t value) {
  return value > 0 && (value & ~(value - 1)) == value;
}

/**
 * Returns the next alignment for a given pointer.
 */
static uint8_t* align_pointer(void* ptr, size_t alignment) {
  intptr_t addr = reinterpret_cast<intptr_t>(ptr);
  if ((addr & (alignment - 1)) == 0) {
    // Already aligned.
    return reinterpret_cast<uint8_t*>(ptr);
  }
  // Bump forward.
  addr = (addr | (alignment - 1)) + 1;
  return reinterpret_cast<uint8_t*>(addr);
}
} // namespace

FileDataLoader::~FileDataLoader() {
  // file_name_ can be nullptr if this instance was moved from, but freeing a
  // null pointer is safe.
  std::free(const_cast<char*>(file_name_));
  // fd_ can be -1 if this instance was moved from, but closing a negative fd is
  // safe (though it will return an error).
  ::close(fd_);
}

Result<FileDataLoader> FileDataLoader::from(
    const char* file_name,
    size_t alignment) {
  ET_CHECK_OR_RETURN_ERROR(
      is_power_of_2(alignment),
      InvalidArgument,
      "Alignment %zu is not a power of 2",
      alignment);

  // Use open() instead of fopen() to avoid the layer of buffering that
  // fopen() does. We will be reading large portions of the file in one shot,
  // so buffering does not help.
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

  return FileDataLoader(fd, file_size, alignment, file_name_copy);
}

namespace {
/**
 * FreeableBuffer::FreeFn-compatible callback.
 *
 * `context` is actually a ptrdiff_t value (not a pointer) that contains the
 * offset in bytes between `data` and the actual pointer to free.
 */
void FreeSegment(void* context, void* data, ET_UNUSED size_t size) {
  ptrdiff_t offset = reinterpret_cast<ptrdiff_t>(context);
  ET_DCHECK_MSG(offset >= 0, "Unexpected offset %ld", (long int)offset);
  std::free(static_cast<uint8_t*>(data) - offset);
}
} // namespace

Result<FreeableBuffer> FileDataLoader::load(
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
      file_name_,
      offset,
      size,
      file_size_);

  // Don't bother allocating/freeing for empty segments.
  if (size == 0) {
    return FreeableBuffer(nullptr, 0, /*free_fn=*/nullptr);
  }

  // Allocate memory for the FreeableBuffer.
  size_t alloc_size = size;
  if (alignment_ > alignof(std::max_align_t)) {
    // malloc() will align to smaller values, but we must manually align to
    // larger values.
    alloc_size += alignment_;
  }
  void* buffer = std::malloc(alloc_size);
  if (buffer == nullptr) {
    ET_LOG(
        Error,
        "Reading from %s at offset %zu: malloc(%zd) failed",
        file_name_,
        offset,
        size);
    return Error::MemoryAllocationFailed;
  }

  // Align.
  void* aligned_buffer = align_pointer(buffer, alignment_);

  // Assert that the alignment didn't overflow the buffer.
  ET_DCHECK_MSG(
      reinterpret_cast<uintptr_t>(aligned_buffer) + size <=
          reinterpret_cast<uintptr_t>(buffer) + alloc_size,
      "aligned_buffer %p + size %zu > buffer %p + alloc_size %zu",
      aligned_buffer,
      size,
      buffer,
      alloc_size);

  auto err = load_into(offset, size, segment_info, aligned_buffer);
  if (err != Error::Ok) {
    // Free `buffer`, which is what malloc() gave us, not `aligned_buffer`.
    std::free(buffer);
    return err;
  }

  // We can't naively free this pointer, since it may not be what malloc() gave
  // us. Pass the offset to the real buffer as context. This is the number of
  // bytes that need to be subtracted from the FreeableBuffer::data() pointer to
  // find the actual pointer to free.
  return FreeableBuffer(
      aligned_buffer,
      size,
      FreeSegment,
      /*free_fn_context=*/
      reinterpret_cast<void*>(
          // Using signed types here because it will produce a signed ptrdiff_t
          // value, though for us it will always be non-negative.
          reinterpret_cast<intptr_t>(aligned_buffer) -
          reinterpret_cast<intptr_t>(buffer)));
}

Result<size_t> FileDataLoader::size() const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  return file_size_;
}

ET_NODISCARD Error FileDataLoader::load_into(
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
      file_name_,
      offset,
      size,
      file_size_);
  ET_CHECK_OR_RETURN_ERROR(
      buffer != nullptr, InvalidArgument, "Provided buffer cannot be null");

  // Read the data into the aligned address.
  size_t needed = size;
  uint8_t* buf = reinterpret_cast<uint8_t*>(buffer);

  // Make a duplicate fd if pread() is not available and we have to seek().
  // Cannot use the standard dup() or fcntl() calls because the returned
  // duplicate will share the underlying file record and affect the original fd
  // when seeking on multiple threads simultaneously.
  const auto dup_fd = ET_HAVE_PREAD ? fd_ : ::open(file_name_, O_RDONLY);

  while (needed > 0) {
    // Reads on macOS will fail with EINVAL if size > INT32_MAX.
    const auto chunk_size = std::min<size_t>(
        needed, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const auto nread =
#if ET_HAVE_PREAD
        ::pread(dup_fd, buf, chunk_size, offset);
#else
        (::lseek(dup_fd, offset, SEEK_SET) == (off_t)-1)
        ? -1
        : ::read(dup_fd, buf, chunk_size);
#endif
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
          file_name_,
          size,
          offset,
          nread == 0 ? "EOF" : strerror(errno));
      if (!ET_HAVE_PREAD) {
        ::close(dup_fd);
      }
      return Error::AccessFailed;
    }
    needed -= nread;
    buf += nread;
    offset += nread;
  }
  if (!ET_HAVE_PREAD) {
    ::close(dup_fd);
  }
  return Error::Ok;
}

} // namespace extension
} // namespace executorch
