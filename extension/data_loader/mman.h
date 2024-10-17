/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file ensures that mman.h compatible functions are defined in the global
// namespace for windows and posix environments.

#pragma once

#include <executorch/runtime/platform/compiler.h>

#ifndef _WIN32

#include <sys/mman.h>
#include <unistd.h>

ET_INLINE long get_os_page_size(){return sysconf(_SC_PAGESIZE)}

#else

#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#include <io.h>

#include <executorch/extension/data_loader/mman_windows.h>

ET_INLINE long get_os_page_size() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  long pagesize = si.dwAllocationGranularity > si.dwPageSize
      ? si.dwAllocationGranularity
      : si.dwPageSize;
  return pagesize;
}

#endif
