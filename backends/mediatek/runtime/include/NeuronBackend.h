/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include "NeuronBufferAllocator.h"
#include "NeuronExecutor.h"
#include "NeuronLog.h"
#include "NeuronPayloadHeader.h"
#include "api/APUWareUtilsLib.h"
#include "api/NeuronAdapter.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace executor {

class NeuronBackend final : public ::executorch::runtime::BackendInterface {
 public:
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override;

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override;

  void destroy(DelegateHandle* handle) const override;

  bool is_available() const override;
};

extern const char kHighAddrKey[];
extern const char kImportForeverKey[];

struct NeuronDelegateSetting {
  bool mHighAddr = false;

  bool mImportForever = false;

  std::string ToRuntimeOption() {
    if (mHighAddr && mImportForever) {
      return "--apusys-config \"{ \\\"high_addr\\\": true, \\\"import_forever\\\": true }\"";
    } else if (mHighAddr) {
      return "--apusys-config \"{ \\\"high_addr\\\": true }\"";
    } else if (mImportForever) {
      return "--apusys-config \"{ \\\"import_forever\\\": true }\"";
    } else {
      return "";
    }
  }
};

class NeuronExecuTorchDelegate {
 public:
  class MemoryCache {
   public:
    template <bool isInput>
    bool IsCached(int i, void* ptr) {
      const auto& cache = isInput ? mInputCache : mOutputCache;
      auto it = cache.find(i);
      return (it != cache.end()) && (ptr == it->second);
    }

    template <bool isInput>
    void UpdateCache(int i, void* ptr) {
      (isInput ? mInputCache[i] : mOutputCache[i]) = ptr;
      return;
    }

   private:
    std::unordered_map<int, void*> mInputCache;

    std::unordered_map<int, void*> mOutputCache;
  };

  NeuronExecuTorchDelegate() {}

  ~NeuronExecuTorchDelegate() {
    mPLock->Stop();
  }

  int LoadCompiledNetwork(
      NeuronPayload payload,
      NeuronDelegateSetting options) {
    mSettings = options;
    auto runtimeOption = mSettings.ToRuntimeOption();
    auto res = mExecutor.LoadFromCompiledNetwork(
        payload.CompiledNetwork,
        payload.Header.DataLen,
        payload.Header.InputCount,
        payload.Header.OutputCount,
        runtimeOption);
    CHECK_NO_ERROR(res);
    CHECK_TRUE(mExecutor.IsValid());
    SummaryIoCounts();
    mPLock = std::unique_ptr<ScopePerformancer>(new ScopePerformancer);
    return NEURON_NO_ERROR;
  }

  Error execute(ET_UNUSED BackendExecutionContext& context, EValue** args)
      const;

 private:
  template <bool isInput>
  bool IsCached(int index, void* ptr) const {
    return mCache.IsCached</*isInput=*/isInput>(index, ptr);
  }

  template <bool isInput>
  void UpdateCache(int index, void* ptr) const {
    mCache.UpdateCache<isInput>(index, ptr);
  }

  int SummaryIoCounts() {
    for (int i = 0;; i++) {
      size_t size = mExecutor.GetInputOutputPaddedSize</*isInput*/ true>(i);
      if (size == 0) {
        break;
      }
      LogInfo("NeuronBackend", "Model input:%d size: %lu", i, size);
      mInputSizes.push_back(size);
    }
    for (int o = 0;; o++) {
      size_t size = mExecutor.GetInputOutputPaddedSize</*isInput*/ false>(o);
      if (size == 0) {
        break;
      }
      LogInfo("NeuronBackend", "Model output:%d size: %lu", o, size);
      mOutputSizes.push_back(size);
    }
    return NEURON_NO_ERROR;
  }

  int HintNeuronBackend(EValue** args) const;

 private:
  std::vector<size_t> mInputSizes;

  std::vector<size_t> mOutputSizes;

  mutable MemoryCache mCache;

  std::unique_ptr<ScopePerformancer> mPLock;

  neuron::NeuronExecutor mExecutor;

  NeuronDelegateSetting mSettings;

  mutable std::unordered_set<const void*> mHasImported;

 private:
  NeuronExecuTorchDelegate(const NeuronExecuTorchDelegate&);

  NeuronExecuTorchDelegate operator=(const NeuronExecuTorchDelegate&);
};

} // namespace executor
} // namespace torch
