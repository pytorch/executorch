/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h> // For ssize_t.
#include <io.h>

#include <windows.h>

inline ssize_t pread(int __fd, void* __buf, size_t __nbytes, size_t __offset) {
  OVERLAPPED overlapped; /* The offset for ReadFile. */
  memset(&overlapped, 0, sizeof(overlapped));
  overlapped.Offset = __offset;
  overlapped.OffsetHigh = __offset >> 32;

  BOOL result; /* The result of ReadFile. */
  DWORD bytes_read; /* The number of bytes read. */
  HANDLE file = (HANDLE)_get_osfhandle(__fd);

  result = ReadFile(file, __buf, __nbytes, &bytes_read, &overlapped);
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

// To avoid conflicts with std::numeric_limits<int32_t>::max() in
// file_data_loader.cpp.
#undef max
