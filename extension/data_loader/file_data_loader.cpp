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
#include <new>
#include <string>

#include <executorch/runtime/platform/compat_unistd.h>
#include <fcntl.h>
#ifndef _WIN32
#include <sys/file.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

#include <c10/util/safe_numerics.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>

// Some platforms (e.g. Xtensa) do not support pread() that we use to read the
// file at different offsets simultaneously from multiple threads not affecting
// each other. We list them below and use a workaround for them.
#if defined(__xtensa__) || defined(__hexagon__)
#define ET_HAVE_PREAD 0
#endif // defined(__xtensa__)

#ifndef ET_HAVE_PREAD
#define ET_HAVE_PREAD 1
#endif // !ET_HAVE_PREAD

using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace executorch {
namespace extension {

namespace {
inline void* et_aligned_alloc(size_t size, std::align_val_t alignment) {
  // Use the nothrow form so allocation failure returns nullptr instead of
  // throwing std::bad_alloc. ExecuTorch is built exception-free and callers
  // (e.g. FileDataLoader::load) check for nullptr and return
  // Error::MemoryAllocationFailed; a throw here would unwind with no landing
  // pad and abort the process.
  return ::operator new(size, alignment, std::nothrow);
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
/**
 * Returns true if the value is an integer power of 2.
 */
static bool is_power_of_2(size_t value) {
  return value > 0 && (value & ~(value - 1)) == value;
}

#ifndef _WIN32
Error write_all(int fd, const void* data, size_t size) {
  const uint8_t* cursor = static_cast<const uint8_t*>(data);
  while (size > 0) {
    const size_t chunk_size = std::min<size_t>(
        size, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const ssize_t written = ::write(fd, cursor, chunk_size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return Error::AccessFailed;
    }
    cursor += written;
    size -= written;
  }
  return Error::Ok;
}

Error copy_from_fd(
    int source_fd,
    int destination_fd,
    size_t offset,
    size_t size) {
  uint8_t buffer[64 * 1024];
  while (size > 0) {
    const size_t chunk_size = std::min(size, sizeof(buffer));
    ssize_t nread = ::pread(source_fd, buffer, chunk_size, offset);
    if (nread < 0 && errno == EINTR) {
      continue;
    }
    if (nread <= 0 ||
        write_all(destination_fd, buffer, static_cast<size_t>(nread)) !=
            Error::Ok) {
      return Error::AccessFailed;
    }
    offset += static_cast<size_t>(nread);
    size -= static_cast<size_t>(nread);
  }
  return Error::Ok;
}
#endif
} // namespace

namespace internal {

Error replace_file_data(
    int fd,
    const char* file_name,
    size_t file_size,
    runtime::Span<const DataLoader::DataChunk> chunks) {
#ifdef _WIN32
  (void)fd;
  (void)file_name;
  (void)file_size;
  (void)chunks;
  return Error::NotSupported;
#else
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0 && file_name != nullptr,
      InvalidState,
      "Uninitialized");

  for (const auto& chunk : chunks) {
    if (chunk.source == DataLoader::DataChunk::Source::Loader) {
      size_t end;
      ET_CHECK_OR_RETURN_ERROR(
          !c10::add_overflows(chunk.offset, chunk.size, &end) &&
              end <= file_size,
          InvalidArgument,
          "Replacement source range is out of bounds");
    } else if (chunk.source == DataLoader::DataChunk::Source::Buffer) {
      ET_CHECK_OR_RETURN_ERROR(
          chunk.data != nullptr || chunk.size == 0,
          InvalidArgument,
          "Replacement buffer is null");
    }
  }

  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    return Error::AccessFailed;
  }

  struct stat source_stat;
  struct stat path_stat;
  if (::fstat(fd, &source_stat) != 0 || ::stat(file_name, &path_stat) != 0 ||
      source_stat.st_dev != path_stat.st_dev ||
      source_stat.st_ino != path_stat.st_ino) {
    ::flock(fd, LOCK_UN);
    return Error::InvalidState;
  }

  std::string temporary_path(file_name);
  temporary_path.append(".rewrite.XXXXXX");
  int temporary_fd = ::mkstemp(temporary_path.data());
  if (temporary_fd < 0) {
    ::flock(fd, LOCK_UN);
    return Error::AccessFailed;
  }
  (void)::fchmod(temporary_fd, source_stat.st_mode & 07777);

  Error result = Error::Ok;
  const uint8_t zeros[4096] = {};
  for (const auto& chunk : chunks) {
    if (chunk.source == DataLoader::DataChunk::Source::Loader) {
      result = copy_from_fd(fd, temporary_fd, chunk.offset, chunk.size);
    } else if (chunk.source == DataLoader::DataChunk::Source::Buffer) {
      result = write_all(temporary_fd, chunk.data, chunk.size);
    } else {
      size_t remaining = chunk.size;
      while (remaining > 0 && result == Error::Ok) {
        const size_t zero_size = std::min(remaining, sizeof(zeros));
        result = write_all(temporary_fd, zeros, zero_size);
        remaining -= zero_size;
      }
    }
    if (result != Error::Ok) {
      break;
    }
  }

  if (result == Error::Ok && ::fsync(temporary_fd) != 0) {
    result = Error::AccessFailed;
  }
  if (::close(temporary_fd) != 0 && result == Error::Ok) {
    result = Error::AccessFailed;
  }
  if (result == Error::Ok &&
      ::rename(temporary_path.c_str(), file_name) != 0) {
    result = Error::AccessFailed;
  }
  if (result != Error::Ok) {
    (void)::unlink(temporary_path.c_str());
  }
  ::flock(fd, LOCK_UN);
  return result;
#endif
}

} // namespace internal

FileDataLoader::~FileDataLoader() {
  // file_name_ can be nullptr if this instance was moved from, but freeing a
  // null pointer is safe.
  et_aligned_free(const_cast<char*>(file_name_), alignment_);
  // fd_ can be -1 if this instance was moved from, but closing a negative fd is
  // safe (though it will return an error).
  if (fd_ == -1) {
    return;
  }
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

  ET_CHECK_OR_RETURN_ERROR(
      file_name != nullptr, InvalidArgument, "File name cannot be empty.");

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
  size_t file_name_len = ::strlen(file_name) + 1;
  char* file_name_copy =
      (char*)et_aligned_alloc(file_name_len, std::align_val_t(alignment));

  if (file_name_copy == nullptr) {
    ET_LOG(Error, "strdup(%s) failed", file_name);
    ::close(fd);
    return Error::MemoryAllocationFailed;
  }
  ::strcpy(file_name_copy, file_name);

  return FileDataLoader(fd, file_size, alignment, file_name_copy);
}

Result<FreeableBuffer> FileDataLoader::load(
    size_t offset,
    size_t size,
    ET_UNUSED const DataLoader::SegmentInfo& segment_info) const {
  ET_CHECK_OR_RETURN_ERROR(
      // Probably had its value moved to another instance.
      fd_ >= 0,
      InvalidState,
      "Uninitialized");
  size_t total_size;
  bool overflow = c10::add_overflows(offset, size, &total_size);
  ET_CHECK_OR_RETURN_ERROR(
      !overflow && total_size <= file_size_,
      InvalidArgument,
      "File %s: offset %zu + size %zu > file_size_ %zu, or overflow detected",
      file_name_,
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
        file_name_,
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
  size_t total_size;
  bool overflow = c10::add_overflows(offset, size, &total_size);
  ET_CHECK_OR_RETURN_ERROR(
      !overflow && total_size <= file_size_,
      InvalidArgument,
      "File %s: offset %zu + size %zu > file_size_ %zu, or overflow detected",
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

Error FileDataLoader::replace_data(runtime::Span<const DataChunk> chunks) {
  const std::lock_guard<std::mutex> lock(replace_mutex_);
  return internal::replace_file_data(fd_, file_name_, file_size_, chunks);
}

} // namespace extension
} // namespace executorch
