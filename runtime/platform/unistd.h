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

#endif // !defined(_WIN64)