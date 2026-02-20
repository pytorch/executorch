/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define QNN_BACKEND "QnnBackend"
#define QNN_RUNTIME_LOG_LEVEL "qnn_runtime_log_level"
#define QNN_RUNTIME_HTP_PERFORMANCE_MODE "qnn_runtime_htp_performance_mode"
#define QNN_RUNTIME_PROFILE_LEVEL "qnn_runtime_profile_level"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// This could be:
// 1. qnn_context_binary
// 2. QnnContextCustomProtocol
// To check if it is custom protocol, users can deserialize the binary using
// QnnCustomProtocol and check the status
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

/// Allocate memory in different way, check qnn document for more details.
enum QnnMemDescriptor { kIon, kCustom };

struct CustomMemTensorInfo {
  void* custom_mem;
  void* tensor_addr;
  size_t pos;
  size_t tensor_bytes;
  uint32_t* shape;
  uint32_t rank;
  executorch::aten::ScalarType dtype;
};

/// Allocate specific tensors (usually graph inputs and outputs) on shared
/// memory. Users are responsible to allocate "enough" tensor bytes, and set
/// alignment as MemoryAllocator::kDefaultAlignment.
/// See runtime/core/memory_allocator.h. The function returns a valid pointer
/// if allocation is successful.
void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment);

/// Add tensor to custom memory with custom type descriptor. Create memory
/// handle to tensor wrapper during execution
void QnnExecuTorchAddCustomMemTensorAddr(void* tensor_addr, void* custom_mem);

/// Free the allocated shared memory.
void QnnExecuTorchFreeCustomMem(void* buffer_ptr);

#ifdef __cplusplus
}
#endif // __cplusplus
