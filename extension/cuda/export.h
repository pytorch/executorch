/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// extension_cuda is a shared library so the caller-stream thread-local has a
// single definition across every shared object in the process; a static copy
// linked into two .so's would create two thread-locals and silently break the
// handshake. These macros export the public symbols from that one library.
#if defined(_WIN32)
#if defined(EXECUTORCH_EXTENSION_CUDA_BUILDING)
#define EXECUTORCH_EXTENSION_CUDA_API __declspec(dllexport)
#else
#define EXECUTORCH_EXTENSION_CUDA_API __declspec(dllimport)
#endif
#else
#define EXECUTORCH_EXTENSION_CUDA_API __attribute__((visibility("default")))
#endif
