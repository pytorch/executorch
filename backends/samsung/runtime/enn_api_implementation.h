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
#include <memory>
#include <mutex>

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

  EnnReturn (*EnnInitialize)(void);
  EnnReturn (*EnnSetPreferencePerfMode)(const uint32_t val);
  EnnReturn (*EnnGetPreferencePerfMode)(uint32_t* val_ptr);
  EnnReturn (*EnnOpenModel)(const char* model_file, EnnModelId* model_id);
  EnnReturn (*EnnOpenModelFromMemory)(
      const char* va,
      const uint32_t size,
      EnnModelId* model_id);
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
  EnnReturn (*EnnExecuteModelWithSessionIdWait)(
      const EnnModelId model_id,
      const int session_id);
  EnnReturn (*EnnBufferCommit)(const EnnModelId model_id);
  EnnReturn (*EnnGetBuffersInfo)(
      const EnnModelId model_id,
      NumberOfBuffersInfo* buffers_info);
  EnnReturn (
      *EnnReleaseBuffers)(EnnBufferPtr* buffers, const int32_t numOfBuffers);

 private:
  static std::mutex instance_mutex_;
  std::atomic_bool initialize_ = false;
  // Pointer to the dlopen libs
  void* libenn_public_api_ = nullptr;
  static std::atomic<int> ref_count_;

  EnnApi() = default;
  bool getInitialize() const;
  Error loadApiLib();
  Error unloadApiLib();
};

typedef EnnReturn (*EnnInitialize_fn)(void);
typedef EnnReturn (*EnnSetPreferencePerfMode_fn)(const uint32_t val);
typedef EnnReturn (*EnnGetPreferencePerfMode_fn)(uint32_t* val_ptr);
typedef EnnReturn (
    *EnnOpenModel_fn)(const char* model_file, EnnModelId* model_id);
typedef EnnReturn (*EnnOpenModelFromMemory_fn)(
    const char* va,
    const uint32_t size,
    EnnModelId* model_id);
typedef EnnReturn (*EnnSetFastIpc_fn)(void);
typedef EnnReturn (*EnnUnsetFastIpc_fn)(void);
typedef EnnReturn (*EnnExecuteModelFastIpc_fn)(
    const EnnModelId model_id,
    int client_sleep_usec);
typedef EnnReturn (*EnnExecuteModel_fn)(const EnnModelId model_id);
typedef EnnReturn (*EnnExecuteModelWithSessionIdAsync_fn)(
    const EnnModelId model_id,
    const int session_id);
typedef EnnReturn (*EnnCloseModel_fn)(const EnnModelId model_id);
typedef EnnReturn (*EnnDeinitialize_fn)(void);
typedef EnnReturn (*EnnAllocateAllBuffers_fn)(
    const EnnModelId model_id,
    EnnBufferPtr** out_buffers,
    NumberOfBuffersInfo* out_buffers_info);
typedef EnnReturn (*EnnAllocateAllBuffersWithSessionId_fn)(
    const EnnModelId model_id,
    EnnBufferPtr** out_buffers,
    NumberOfBuffersInfo* out_buffers_info,
    const int session_id,
    const bool do_commit);
typedef EnnReturn (*EnnExecuteModelWithSessionIdWait_fn)(
    const EnnModelId model_id,
    const int session_id);
typedef EnnReturn (*EnnBufferCommit_fn)(const EnnModelId model_id);
typedef EnnReturn (*EnnGetBuffersInfo_fn)(
    const EnnModelId model_id,
    NumberOfBuffersInfo* buffers_info);
typedef EnnReturn (
    *EnnReleaseBuffers_fn)(EnnBufferPtr* buffers, const int32_t numOfBuffers);

} // namespace enn
} // namespace executor
} // namespace torch
