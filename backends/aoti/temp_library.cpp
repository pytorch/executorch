/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/temp_library.h>
#include <atomic>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <cstdio>
#include <random>
#else // Posix
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#endif

namespace executorch {
namespace backends {
namespace aoti {

executorch::runtime::Result<std::filesystem::path> write_temp_library(
    const void* data,
    size_t size) {
  std::error_code temp_dir_error;
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path(temp_dir_error);
  if (temp_dir_error) {
    ET_LOG(
        Error,
        "Failed to find the temporary directory: %s",
        temp_dir_error.message().c_str());
    return executorch::runtime::Error::AccessFailed;
  }
  // The loader hands back an already loaded library when given the same path
  // again, and libraries are never unloaded, so a path must not repeat within
  // a process.
  static std::atomic<uint64_t> counter{0};
  const std::string prefix = "executorch_aoti_" +
      std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + "_";
  const char* bytes = static_cast<const char*>(data);
  size_t remaining = size;
#ifdef _WIN32
  std::random_device random;
  char suffix[24];
  snprintf(suffix, sizeof(suffix), "%08x%08x.so", random(), random());
  std::filesystem::path path = temp_dir / (prefix + suffix);
  const auto u8 = path.u8string();
  const std::string utf8(u8.begin(), u8.end());
  HANDLE file = CreateFileW(
      path.c_str(),
      GENERIC_WRITE,
      0,
      nullptr,
      CREATE_NEW,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    ET_UNUSED const DWORD err = GetLastError();
    ET_LOG(Error, "Failed to create %s with error %lu", utf8.c_str(), err);
    return executorch::runtime::Error::AccessFailed;
  }
  // WriteFile takes a 32-bit length, so write in chunks.
  constexpr DWORD kMaxChunk = 1u << 30;
  DWORD err = ERROR_SUCCESS;
  while (remaining > 0 && err == ERROR_SUCCESS) {
    const DWORD chunk =
        remaining > kMaxChunk ? kMaxChunk : static_cast<DWORD>(remaining);
    DWORD written = 0;
    if (!WriteFile(file, bytes, chunk, &written, nullptr)) {
      err = GetLastError();
    } else if (written == 0) {
      err = ERROR_WRITE_FAULT;
    } else {
      bytes += written;
      remaining -= written;
    }
  }
  if (!CloseHandle(file) && err == ERROR_SUCCESS) {
    err = GetLastError();
  }
  if (err != ERROR_SUCCESS) {
    DeleteFileW(path.c_str());
    ET_LOG(Error, "Failed to write %s with error %lu", utf8.c_str(), err);
    return executorch::runtime::Error::AccessFailed;
  }
  ET_LOG(Info, "Wrote %zu bytes to %s", size, utf8.c_str());
  return path;
#else
  // mkostemps replaces the Xs with random characters and creates the file with
  // O_EXCL and mode 0600, retrying if the name is taken.
  std::string path = (temp_dir / (prefix + "XXXXXX.so")).string();
  const int fd = mkostemps(path.data(), /*suffixlen=*/3, O_CLOEXEC);
  if (fd == -1) {
    ET_UNUSED const int err = errno;
    ET_LOG(Error, "Failed to create %s: %s", path.c_str(), strerror(err));
    return executorch::runtime::Error::AccessFailed;
  }
  int err = 0;
  while (remaining > 0 && err == 0) {
    const ssize_t written = write(fd, bytes, remaining);
    if (written > 0) {
      bytes += written;
      remaining -= static_cast<size_t>(written);
    } else if (written == 0) {
      err = EIO;
    } else if (errno != EINTR) {
      err = errno;
    }
  }
  if (close(fd) != 0 && err == 0) {
    err = errno;
  }
  if (err != 0) {
    ET_LOG(Error, "Failed to write %s: %s", path.c_str(), strerror(err));
    unlink(path.c_str());
    return executorch::runtime::Error::AccessFailed;
  }
  ET_LOG(Info, "Wrote %zu bytes to %s", size, path.c_str());
  // The caller only takes over the file once this returns, so remove it here
  // if building the path throws.
  try {
    return std::filesystem::path(path);
  } catch (...) {
    unlink(path.c_str());
    throw;
  }
#endif
}

} // namespace aoti
} // namespace backends
} // namespace executorch
