
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
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else // Posix
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#endif

namespace executorch {
namespace backends {
namespace cuda {

executorch::runtime::Result<void*> load_library(
    const std::filesystem::path& path) {
#ifdef _WIN32
  std::string utf8 = path.u8string();
  auto lib_handle = LoadLibrary(utf8.c_str());
  if (lib_handle == NULL) {
    ET_LOG(
        Error,
        "Failed to load %s with error: %lu",
        utf8.c_str(),
        GetLastError());
    return executorch::runtime::Error::AccessFailed;
  }

#else
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

int32_t get_process_id() {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return getpid();
#endif
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
