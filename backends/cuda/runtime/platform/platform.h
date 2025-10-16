
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace executorch {
namespace backends {
namespace cuda {

executorch::runtime::Result<void*> load_library(const std::string& path) {
#ifdef _WIN32
  auto lib_handle = LoadLibrary(path.c_str());
  if (lib_handle == NULL) {
    ET_LOG(
        Error,
        "Failed to load %s with error: %lu",
        path.c_str(),
        GetLastError());
    return Error::AccessFailed;
  }

#else
  void* lib_handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (lib_handle == nullptr) {
    ET_LOG(Error, "Failed to load %s with error: %s", path.c_str(), dlerror());
    return Error::AccessFailed;
  }
#endif
  return (void*)lib_handle;
}

executorch::runtime::Error close_library(void* lib_handle) {
#ifdef _WIN32
  if (!FreeLibrary((HMODULE)lib_handle)) {
    printf("FreeLibrary failed with error %lu\n", GetLastError());
    return Error::Internal;
  }
#else
  if (dlclose(lib_handle) != 0) {
    ET_LOG(Error, "dlclose failed: %s\n", dlerror());
    return Error::Internal;
  }
#endif
  return Error::Ok;
}

executorch::runtime::Result<void*> get_function(void* lib_handle, const std::string& fn_name) {
#ifdef _WIN32
  auto fn = GetProcAddress((HMODULE)lib_handle, fn_name.c_str());
  if (!fn) {
    ET_LOG(
        Error,
        "Failed loading symbol %s with error %lu\n",
        fn_name.c_str(),
        GetLastError());
    return Error::Internal;
  }
#else
  auto fn = dlsym(lib_handle, fn_name.c_str());
  if (fn == nullptr) {
    ET_LOG(
        Error,
        "Failed loading symbol %s with error %s\n",
        fn_name.c_str(),
        dlerror());
    return Error::Internal;
  }
#endif

  return (void*)fn; // This I think is technically ub on windows. We should
                    // probably explicitly pack the bytes.
}

int32_t get_process_id(void* lib_handle) {
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

}

} // cuda
} // backends
} // executorch
