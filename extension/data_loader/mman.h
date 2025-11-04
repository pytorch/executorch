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
#include <sys/stat.h>
#include <cstdint>

#ifndef _WIN32

#include <sys/mman.h>
#include <unistd.h>

ET_INLINE size_t get_os_page_size() {
  return sysconf(_SC_PAGESIZE);
}

/**
 * Platform-specific file stat function.
 */
ET_INLINE int get_file_stat(int fd, size_t* out_size) {
  struct stat st;
  int err = ::fstat(fd, &st);
  if (err >= 0) {
    *out_size = static_cast<size_t>(st.st_size);
  }
  return err;
}

/**
 * Platform-specific mmap offset type conversion.
 */
ET_INLINE off_t get_mmap_offset(size_t offset) {
  return static_cast<off_t>(offset);
}

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

/**
 * Platform-specific file stat function.
 */
ET_INLINE int get_file_stat(int fd, size_t* out_size) {
  struct _stat64 st;
  int err = ::_fstat64(fd, &st);
  if (err >= 0) {
    *out_size = static_cast<size_t>(st.st_size);
  }
  return err;
}

/**
 * Platform-specific mmap offset type conversion.
 */
ET_INLINE uint64_t get_mmap_offset(size_t offset) {
  return static_cast<uint64_t>(offset);
}

#endif
