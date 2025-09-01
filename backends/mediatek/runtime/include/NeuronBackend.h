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
#include "executorch/runtime/core/exec_aten/util/dim_order_util.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace neuron {

using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class NeuronSharedWeights {
 public:
  explicit NeuronSharedWeights(const FreeableBuffer& shared_weights_buffer) {
    auto& buffer_allocator = GET_NEURON_ALLOCATOR;
    nbytes_ = shared_weights_buffer.size();
    data_ = buffer_allocator.Allocate(nbytes_);
    ET_CHECK_MSG(
        data_ != nullptr,
        "Error: Failed to allocate memory for shared weights of size %zu",
        nbytes_);
    std::memcpy(data_, shared_weights_buffer.data(), nbytes_);
  }

  explicit NeuronSharedWeights(FreeableBuffer&& shared_weights_buffer)
      : NeuronSharedWeights(shared_weights_buffer) {
    shared_weights_buffer.Free();
  }

  ~NeuronSharedWeights() {
    if (data_ == nullptr || nbytes_ == 0) {
      return;
    }
    auto& buffer_allocator = GET_NEURON_ALLOCATOR;
    buffer_allocator.RemoveBuffer(data_);
  }

  void* data() const {
    return data_;
  }

  size_t size() const {
    return nbytes_;
  }

 private:
  void* data_ = nullptr;
  size_t nbytes_ = 0;
};

class NeuronBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec>
          compile_specs) const override;

  ::executorch::runtime::Error execute(
      ET_UNUSED ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::EValue** args) const override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  mutable std::unordered_map<std::string, std::weak_ptr<NeuronSharedWeights>>
      neuron_shared_weights_cache_;
};

extern const char kHighAddrKey[];
extern const char kImportForeverKey[];

struct NeuronDelegateSetting {
  bool mHighAddr = false;

  bool mImportForever = false;

  bool mSharedWeights = false;

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
  struct InputOutputInfo {
    void* data_ptr;
    size_t size;

    InputOutputInfo(void* ptr, size_t sz) : data_ptr(ptr), size(sz) {}
  };

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
        mSettings.mSharedWeights ? payload.Header.InputCount + 1
                                 : payload.Header.InputCount,
        payload.Header.OutputCount,
        runtimeOption);
    CHECK_NO_ERROR(res);
    CHECK_TRUE(mExecutor.IsValid());
    SummarizeIoSizes(payload.Header.InputCount, payload.Header.OutputCount);
    mPLock = std::unique_ptr<ScopePerformancer>(new ScopePerformancer);
    return NEURON_NO_ERROR;
  }

  int SetSharedWeights(std::shared_ptr<NeuronSharedWeights> sharedWeights) {
    neuron_shared_weights_.push_back(sharedWeights);
    return NEURON_NO_ERROR;
  }

  ::executorch::runtime::Error execute(
      ET_UNUSED ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::EValue** args) const;

 private:
  template <bool isInput>
  bool IsCached(int index, void* ptr) const {
    return mCache.IsCached</*isInput=*/isInput>(index, ptr);
  }

  template <bool isInput>
  void UpdateCache(int index, void* ptr) const {
    mCache.UpdateCache<isInput>(index, ptr);
  }

  int SummarizeIoSizes(uint32_t input_count, uint32_t output_count) {
    for (int i = 0; i < input_count; i++) {
      size_t size = mExecutor.GetInputOutputPaddedSize</*isInput*/ true>(i);
      if (size == 0) {
        LogWarn("NeuronBackend", "Model input:%d got size: %lu", i, size);
      }
      LogInfo("NeuronBackend", "Model input:%d size: %lu", i, size);
      mInputSizes.push_back(size);
    }
    for (int o = 0; o < output_count; o++) {
      size_t size = mExecutor.GetInputOutputPaddedSize</*isInput*/ false>(o);
      if (size == 0) {
        LogWarn("NeuronBackend", "Model output:%d got size: %lu", o, size);
      }
      LogInfo("NeuronBackend", "Model output:%d size: %lu", o, size);
      mOutputSizes.push_back(size);
    }
    return NEURON_NO_ERROR;
  }

  int CheckDimOrder(EValue** args) const {
    size_t data_input_count = mInputSizes.size();
    for (int i = 0; i < data_input_count; i++) {
      auto tensor_in = args[i]->toTensor();
      LogInfo("NeuronBackend", "Checking dim order for input %d", i);
      if (!runtime::is_contiguous_dim_order(
              tensor_in.dim_order().data(), tensor_in.dim())) {
        return NEURON_BAD_DATA;
      }
    }

    return NEURON_NO_ERROR;
  }

  int PrepareInputsOuputs(EValue** args) const {
    bool has_shared_weights_input = neuron_shared_weights_.size() > 0;

    size_t data_input_count = mInputSizes.size();
    size_t data_output_count = mOutputSizes.size();

    mPreparedInputs.clear();
    mPreparedOutputs.clear();
    mPreparedInputs.reserve(data_input_count);
    mPreparedOutputs.reserve(data_output_count);

    // Prepare input data
    for (int i = 0; i < data_input_count; i++) {
      auto tensor_in = args[i]->toTensor();
      auto data_ptr = tensor_in.data_ptr();
      auto data_size = tensor_in.nbytes();
      mPreparedInputs.push_back(InputOutputInfo{data_ptr, data_size});
    }

    // Prepare shared weights if any as the last model inputs
    if (has_shared_weights_input) {
      for (const auto& shared_weights : neuron_shared_weights_) {
        mPreparedInputs.push_back(
            InputOutputInfo{shared_weights->data(), shared_weights->size()});
      }
    }

    // Prepare output data
    for (int o = data_input_count; o < data_input_count + data_output_count;
         o++) {
      auto tensor_out = args[o]->toTensor();
      auto data_ptr = tensor_out.data_ptr();
      auto data_size = tensor_out.nbytes();
      mPreparedOutputs.push_back(InputOutputInfo{data_ptr, data_size});
    }

    return NEURON_NO_ERROR;
  }

  int HintNeuronBackend(::executorch::runtime::EValue** args) const;

 private:
  std::vector<size_t> mInputSizes;

  std::vector<size_t> mOutputSizes;

  mutable std::vector<InputOutputInfo> mPreparedInputs;

  mutable std::vector<InputOutputInfo> mPreparedOutputs;

  mutable MemoryCache mCache;

  std::unique_ptr<ScopePerformancer> mPLock;

  neuron::NeuronExecutor mExecutor;

  NeuronDelegateSetting mSettings;

  mutable std::unordered_set<const void*> mHasImported;

  mutable std::vector<std::shared_ptr<NeuronSharedWeights>>
      neuron_shared_weights_;

 private:
  NeuronExecuTorchDelegate(const NeuronExecuTorchDelegate&);

  NeuronExecuTorchDelegate operator=(const NeuronExecuTorchDelegate&);
};

} // namespace neuron
} // namespace backends
} // namespace executorch
