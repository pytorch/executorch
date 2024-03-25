/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
typedef struct {
  /// qnn_context_binary_blob
  void* buffer;
  /// number of bytes of buffer
  uint64_t nbytes;
} QnnExecuTorchContextBinary;

// clang-format off
#define QNN_EXECUTORCH_CONTEXT_BINARY    \
  {                                      \
    nullptr,        /*buffer*/           \
    0,              /*nbytes*/           \
  }
// clang-format on

/// Allocate specific tensors (usually graph inputs and outputs) on shared
/// memory. Users are responsible to allocate "enough" tensor bytes, and set
/// alignment as MemoryAllocator::kDefaultAlignment.
/// See runtime/core/memory_allocator.h. The function returns a valid pointer
/// if allocation is successful.
void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment);

/// Free the allocated shared memory.
void QnnExecuTorchFreeCustomMem(void* buffer_ptr);

#ifdef __cplusplus
}
#endif // __cplusplus
