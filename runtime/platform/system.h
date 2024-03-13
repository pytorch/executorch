/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Platform abstraction layer to allow individual host OS to override
 * symbols in ExecuTorch. PAL functions are defined as C functions so an
 * implementer can use C in lieu of C++.
 */
#pragma once

/**
 * To enable dynamic linking debugging capability on UNIX-like OS. If enabled
 * and see an error like: `undefined symbol: dladdr`, install `libdl` to fix.
 */
#if defined(ET_USE_LIBDL)
#include <dlfcn.h>
#endif

static constexpr const char* DYNAMIC_LIBRARY_NOT_SUPPORTED = "NOT_SUPPORTED";
static constexpr const char* DYNAMIC_LIBRARY_NOT_FOUND = "NOT_FOUND";

extern "C" {

/**
 * Return shared library .
 *
 * @param[in] addr Address to the symbol we are looking for in shared libraries.
 * @retval The path to the shared library containing the symbol.
 */
inline const char* et_pal_get_shared_library_name(const void* addr) {
#if defined(ET_USE_LIBDL)
  Dl_info info;
  if (dladdr(addr, &info) && info.dli_fname) {
    return info.dli_fname;
  } else {
    return DYNAMIC_LIBRARY_NOT_FOUND;
  }
#endif
  return DYNAMIC_LIBRARY_NOT_SUPPORTED;
}

} // extern "C"
