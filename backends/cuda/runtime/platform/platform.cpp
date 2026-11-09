
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/platform/platform.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#include <cstdio>
#include <random>
#include <string>
#else // Posix
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#endif

#ifdef _WIN32
namespace {
std::string format_win_error(DWORD err) {
  LPSTR buffer = nullptr;
  const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
  const DWORD size = FormatMessageA(
      flags,
      nullptr,
      err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&buffer),
      0,
      nullptr);
  if (size == 0 || buffer == nullptr) {
    return "unknown error";
  }
  std::string message(buffer, size);
  LocalFree(buffer);
  // Trim trailing newlines.
  while (!message.empty() &&
         (message.back() == '\n' || message.back() == '\r')) {
    message.pop_back();
  }
  return message;
}
} // namespace
#endif

namespace executorch {
namespace backends {
namespace cuda {

executorch::runtime::Result<void*> load_library(
    const std::filesystem::path& path) {
#ifdef _WIN32
  auto lib_handle = LoadLibraryW(path.c_str());
  if (lib_handle == NULL) {
    const DWORD err = GetLastError();
    // u8string() returns std::u8string under C++20, so copy into std::string.
    const auto u8 = path.u8string();
    const std::string utf8(u8.begin(), u8.end());
    ET_LOG(
        Error,
        "Failed to load %s with error %lu: %s",
        utf8.c_str(),
        err,
        format_win_error(err).c_str());
    return executorch::runtime::Error::AccessFailed;
  }

#else
  // Before loading the delegate .so, we need to ensure symbols from the current
  // process (e.g. this backend library) are globally visible. Python loads
  // modules with RTLD_LOCAL by default, so we re-open the current module with
  // RTLD_GLOBAL | RTLD_NOLOAD to promote its symbols to global visibility.
  // This allows the delegate .so to resolve symbols like aoti_torch_dtype_*.
  static std::once_flag symbols_promoted_flag;
  std::call_once(symbols_promoted_flag, []() {
    Dl_info info;
    // Get info about a symbol we know exists in this backend library
    if (dladdr((void*)&load_library, &info) && info.dli_fname) {
      // Re-open with RTLD_GLOBAL | RTLD_NOLOAD to promote symbols. Clear any
      // earlier error first, so dlerror() below reports only this call's.
      dlerror();
      void* handle =
          dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
      if (!handle) {
        const char* error = dlerror();
        if (error != nullptr) {
          ET_LOG(Error, "Failed to promote symbols: %s", error);
        } else {
          // dlopen found no library to re-open and set no error. That is the
          // case when this backend is linked into the executable, whose
          // symbols are global already.
          ET_LOG(
              Debug,
              "Not promoting symbols: %s is not loaded as a shared library",
              info.dli_fname);
        }
      } else {
        // Close the handle after successful promotion
        dlclose(handle);
      }
    }
  });

  std::string path_str = path.string();
  void* lib_handle = dlopen(path_str.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (lib_handle == nullptr) {
    ET_LOG(
        Error, "Failed to load %s with error: %s", path_str.c_str(), dlerror());
    return executorch::runtime::Error::AccessFailed;
  }
#endif
  return (void*)lib_handle;
}

executorch::runtime::Error close_library(void* lib_handle) {
#ifdef _WIN32
  if (!FreeLibrary((HMODULE)lib_handle)) {
    printf("FreeLibrary failed with error %lu\n", GetLastError());
    return executorch::runtime::Error::Internal;
  }
#else
  if (dlclose(lib_handle) != 0) {
    ET_LOG(Error, "dlclose failed: %s\n", dlerror());
    return executorch::runtime::Error::Internal;
  }
#endif
  return executorch::runtime::Error::Ok;
}

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
  const std::string prefix = "executorch_cuda_" +
      std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + "_";
  const char* bytes = static_cast<const char*>(data);
  size_t remaining = size;
  bool failed = false;
#ifdef _WIN32
  std::random_device random;
  char suffix[24];
  snprintf(suffix, sizeof(suffix), "%08x%08x.so", random(), random());
  const std::filesystem::path path = temp_dir / (prefix + suffix);
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
    const DWORD err = GetLastError();
    ET_LOG(
        Error,
        "Failed to create %s with error %lu: %s",
        utf8.c_str(),
        err,
        format_win_error(err).c_str());
    return executorch::runtime::Error::AccessFailed;
  }
  // WriteFile takes a 32-bit length, so write in chunks.
  constexpr DWORD kMaxChunk = 1u << 30;
  while (remaining > 0 && !failed) {
    const DWORD chunk =
        remaining > kMaxChunk ? kMaxChunk : static_cast<DWORD>(remaining);
    DWORD written = 0;
    if (WriteFile(file, bytes, chunk, &written, nullptr) && written > 0) {
      bytes += written;
      remaining -= written;
    } else {
      failed = true;
    }
  }
  if (!CloseHandle(file)) {
    failed = true;
  }
  if (failed) {
    const DWORD err = GetLastError();
    ET_LOG(
        Error,
        "Failed to write %s with error %lu: %s",
        utf8.c_str(),
        err,
        format_win_error(err).c_str());
    DeleteFileW(path.c_str());
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
    ET_LOG(Error, "Failed to create %s: %s", path.c_str(), strerror(errno));
    return executorch::runtime::Error::AccessFailed;
  }
  while (remaining > 0 && !failed) {
    const ssize_t written = write(fd, bytes, remaining);
    if (written > 0) {
      bytes += written;
      remaining -= static_cast<size_t>(written);
    } else if (written == 0 || errno != EINTR) {
      failed = true;
    }
  }
  if (close(fd) != 0) {
    failed = true;
  }
  if (failed) {
    ET_LOG(Error, "Failed to write %s: %s", path.c_str(), strerror(errno));
    unlink(path.c_str());
    return executorch::runtime::Error::AccessFailed;
  }
  ET_LOG(Info, "Wrote %zu bytes to %s", size, path.c_str());
  return std::filesystem::path(path);
#endif
}

executorch::runtime::Result<void*> get_function(
    void* lib_handle,
    const std::string& fn_name) {
#ifdef _WIN32
  auto fn = GetProcAddress((HMODULE)lib_handle, fn_name.c_str());
  if (!fn) {
    ET_LOG(
        Error,
        "Failed loading symbol %s with error %lu\n",
        fn_name.c_str(),
        GetLastError());
    return executorch::runtime::Error::Internal;
  }
#else
  auto fn = dlsym(lib_handle, fn_name.c_str());
  if (fn == nullptr) {
    ET_LOG(
        Error,
        "Failed loading symbol %s with error %s\n",
        fn_name.c_str(),
        dlerror());
    return executorch::runtime::Error::Internal;
  }
#endif

  return (void*)fn; // This I think is technically ub on windows. We should
                    // probably explicitly pack the bytes.
}

void* aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

} // namespace cuda
} // namespace backends
} // namespace executorch
