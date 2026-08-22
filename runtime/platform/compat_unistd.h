/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * unistd.h related macros for POSIX/Windows compatibility.
 */
#pragma once

#if defined(_WIN32) && !defined(_WIN64)
#error \
    "You're trying to build ExecuTorch with a too old version of Windows. We need Windows 64-bit."
#endif

#if !defined(_WIN64)
#include <unistd.h>
#else
#include <io.h>
#define O_RDONLY _O_RDONLY
#define open _open
#define close _close
#define read _read
#define write _write
#define stat _stat64
#define fstat _fstat64
#define off_t _off_t
#define lseek _lseeki64

#include <executorch/runtime/platform/compiler.h> // For ssize_t.
#include <windows.h>
// To avoid conflicts with std::numeric_limits<int32_t>::max() in
// file_data_loader.cpp.
#undef max

inline ssize_t pread(int fd, void* buf, size_t nbytes, size_t offset) {
  OVERLAPPED overlapped; /* The offset for ReadFile. */
  memset(&overlapped, 0, sizeof(overlapped));
  overlapped.Offset = offset;
  overlapped.OffsetHigh = offset >> 32;

  BOOL result; /* The result of ReadFile. */
  DWORD bytes_read; /* The number of bytes read. */
  HANDLE file = (HANDLE)_get_osfhandle(fd);

  result = ReadFile(file, buf, nbytes, &bytes_read, &overlapped);
  DWORD error = GetLastError();
  if (!result) {
    if (error == ERROR_IO_PENDING) {
      result = GetOverlappedResult(file, &overlapped, &bytes_read, TRUE);
      if (!result) {
        error = GetLastError();
      }
    }
  }
  if (!result) {
    // Translate error into errno.
    switch (error) {
      case ERROR_HANDLE_EOF:
        errno = 0;
        break;
      default:
        errno = EIO;
        break;
    }
    return -1;
  }
  return bytes_read;
}

#endif // !defined(_WIN64)