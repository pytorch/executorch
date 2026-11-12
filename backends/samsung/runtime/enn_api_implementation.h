/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once
#include <executorch/backends/samsung/runtime/enn_type.h>
#include <executorch/runtime/core/error.h>

#include <atomic>
#include <cstdint>

namespace torch {
namespace executor {
namespace enn {

class EnnApi {
 public:
  EnnApi(const EnnApi&) = delete;
  EnnApi& operator=(const EnnApi&) = delete;
  EnnApi(EnnApi&&) = delete;
  EnnApi& operator=(EnnApi&&) = delete;
  ~EnnApi();

  static EnnApi* getEnnApiInstance();
  bool isInitialized() const;

  EnnReturn (*EnnInitialize)(void);
  EnnReturn (*EnnSetPreferencePerfMode)(const uint32_t val);
  EnnReturn (*EnnGetPreferencePerfMode)(uint32_t* val_ptr);
  EnnReturn (*EnnOpenModel)(const char* model_file, EnnModelId* model_id);
  EnnReturn (*EnnOpenModelFromMemory)(
      const char* va,
      const uint32_t size,
      EnnModelId* model_id);
  EnnReturn (*EnnOpenModelFromFd)(int _fd, EnnModelId* model_id);
  EnnReturn (*EnnSetFastIpc)(void);
  EnnReturn (*EnnUnsetFastIpc)(void);
  EnnReturn (*EnnExecuteModelFastIpc)(
      const EnnModelId model_id,
      int client_sleep_usec);
  EnnReturn (*EnnExecuteModel)(const EnnModelId model_id);
  EnnReturn (*EnnExecuteModelWithSessionIdAsync)(
      const EnnModelId model_id,
      const int session_id);
  EnnReturn (*EnnCloseModel)(const EnnModelId model_id);
  EnnReturn (*EnnDeinitialize)(void);
  EnnReturn (*EnnAllocateAllBuffers)(
      const EnnModelId model_id,
      EnnBufferPtr** out_buffers,
      NumberOfBuffersInfo* out_buffers_info);
  EnnReturn (*EnnAllocateAllBuffersWithSessionId)(
      const EnnModelId model_id,
      EnnBufferPtr** out_buffers,
      NumberOfBuffersInfo* out_buffers_info,
      const int session_id,
      const bool do_commit);
  EnnReturn (*EnnGenerateBufferSpaceDev)(
      const EnnModelId model_id,
      const int session_count);
  EnnReturn (*EnnExecuteModelWithSessionIdWait)(
      const EnnModelId model_id,
      const int session_id);
  EnnReturn (*EnnBufferCommit)(const EnnModelId model_id);
  EnnReturn (*EnnGetBuffersInfo)(
      const EnnModelId model_id,
      NumberOfBuffersInfo* buffers_info);
  EnnReturn (
      *EnnReleaseBuffers)(EnnBufferPtr* buffers, const int32_t numOfBuffers);
  EnnReturn (*EnnCreateBuffer)(
      const uint32_t req_size,
      const uint32_t ion_flag,
      EnnBufferPtr* out);
  EnnReturn (*EnnReleaseBuffer)(EnnBufferPtr buf);
  EnnReturn (
      *EnnGetFileDescriptorFromEnnBuffer)(EnnBufferPtr buffer, int32_t* fd);

 private:
  std::atomic_bool initialize_ = false;
  // Pointer to the dlopen libs
  void* libenn_public_api_ = nullptr;

  EnnApi();
  Error loadApiLib();
  Error unloadApiLib();
};

using EnnInitialize_fn = EnnReturn (*)(void);
using EnnSetPreferencePerfMode_fn = EnnReturn (*)(const uint32_t val);
using EnnGetPreferencePerfMode_fn = EnnReturn (*)(uint32_t* val_ptr);
using EnnOpenModel_fn =
    EnnReturn (*)(const char* model_file, EnnModelId* model_id);
using EnnOpenModelFromMemory_fn = EnnReturn (*)(
    const char* va,
    const uint32_t size,
    EnnModelId* model_id);
using EnnSetFastIpc_fn = EnnReturn (*)(void);
using EnnUnsetFastIpc_fn = EnnReturn (*)(void);
using EnnExecuteModelFastIpc_fn = EnnReturn (*)(
    const EnnModelId model_id,
    int client_sleep_usec);
using EnnExecuteModel_fn = EnnReturn (*)(const EnnModelId model_id);
using EnnExecuteModelWithSessionIdAsync_fn = EnnReturn (*)(
    const EnnModelId model_id,
    const int session_id);
using EnnCloseModel_fn = EnnReturn (*)(const EnnModelId model_id);
using EnnDeinitialize_fn = EnnReturn (*)(void);
using EnnAllocateAllBuffers_fn = EnnReturn (*)(
    const EnnModelId model_id,
    EnnBufferPtr** out_buffers,
    NumberOfBuffersInfo* out_buffers_info);
using EnnAllocateAllBuffersWithSessionId_fn = EnnReturn (*)(
    const EnnModelId model_id,
    EnnBufferPtr** out_buffers,
    NumberOfBuffersInfo* out_buffers_info,
    const int session_id,
    const bool do_commit);
using EnnGenerateBufferSpaceDev_fn =
    EnnReturn (*)(const EnnModelId model_id, const int session_count);
using EnnExecuteModelWithSessionIdWait_fn = EnnReturn (*)(
    const EnnModelId model_id,
    const int session_id);
using EnnBufferCommit_fn = EnnReturn (*)(const EnnModelId model_id);
using EnnGetBuffersInfo_fn = EnnReturn (*)(
    const EnnModelId model_id,
    NumberOfBuffersInfo* buffers_info);
using EnnReleaseBuffers_fn =
    EnnReturn (*)(EnnBufferPtr* buffers, const int32_t numOfBuffers);
using EnnCreateBuffer_fn = EnnReturn (*)(
    const uint32_t req_size,
    const uint32_t ion_flag,
    EnnBufferPtr* out);
using EnnReleaseBuffer_fn = EnnReturn (*)(EnnBufferPtr buf);
using EnnGetFileDescriptorFromEnnBuffer_fn =
    EnnReturn (*)(EnnBufferPtr buffer, int32_t* fd);
using EnnOpenModelFromFd_fn = EnnReturn (*)(int _fd, EnnModelId* model_id);

} // namespace enn
} // namespace executor
} // namespace torch
