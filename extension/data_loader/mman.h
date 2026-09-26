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

#include <fcntl.h>
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

/**
 * Hint the kernel to prefetch pages eagerly and to optimize for sequential
 * reads. Intended to reduce page-fault stutter during model initialization
 * when the caller does not want to mlock the pages into RAM.
 *
 * MADV_WILLNEED / MADV_SEQUENTIAL are absent on some POSIX libcs (e.g. the
 * Hexagon DSP toolchain).
 */
ET_INLINE void madvise_pages_willneed_sequential(void* addr, size_t len) {
#ifdef MADV_WILLNEED
  ::madvise(addr, len, MADV_WILLNEED);
#endif
#ifdef MADV_SEQUENTIAL
  ::madvise(addr, len, MADV_SEQUENTIAL);
#endif
}

/**
 * On Apple platforms, schedule kernel read-ahead on the file descriptor itself
 * via fcntl(F_RDADVISE). This is more aggressive than madvise for cold starts:
 * it brings pages into the unified buffer cache so first-touch faults are
 * serviced from RAM instead of storage. No-op on non-Apple POSIX platforms.
 */
ET_INLINE void fcntl_rdadvise_apple(int fd, size_t file_size) {
#if defined(__APPLE__)
  struct radvisory advice;
  advice.ra_offset = 0;
  advice.ra_count = static_cast<int>(file_size);
  ::fcntl(fd, F_RDADVISE, &advice);
#else
  (void)fd;
  (void)file_size;
#endif
}

#else

#ifndef NOMINMAX
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <windows.h>
#endif
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

/**
 * No-op on Windows: there is no direct equivalent to madvise(MADV_WILLNEED |
 * MADV_SEQUENTIAL) and the existing mman_windows shim does not implement one.
 */
ET_INLINE void madvise_pages_willneed_sequential(void* addr, size_t len) {
  (void)addr;
  (void)len;
}

/**
 * No-op on Windows: F_RDADVISE is an Apple-specific fcntl command.
 */
ET_INLINE void fcntl_rdadvise_apple(int fd, size_t file_size) {
  (void)fd;
  (void)file_size;
}

#endif
