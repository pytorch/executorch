/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "NeuronBackend.h"
#include "NeuronBufferAllocator.h"
#include "NeuronLog.h"
#include "NeuronPayloadHeader.h"
#include "api/NeuronAdapter.h"

#include <executorch/runtime/executor/pte_data_map.h>
#include "executorch/runtime/core/error.h"

#include <algorithm>
#include <memory>
#include <new>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace neuron {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

const char kHighAddrKey[] = "HighAddr";
const char kImportForeverKey[] = "ImportForever";
const char kSharedWeightsKey[] = "ExtractSharedBlobKey";

Result<DelegateHandle*> NeuronBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  NeuronDelegateSetting setting;
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  NeuronExecuTorchDelegate* delegate =
      runtime_allocator->allocateInstance<NeuronExecuTorchDelegate>();
  if (delegate == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  new (delegate) NeuronExecuTorchDelegate();

  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, kHighAddrKey) == 0) {
      setting.mHighAddr = *static_cast<char*>(compile_spec.value.buffer);
      LogInfo("NeuronBackend", "IsHighAddr Enable : %d", setting.mHighAddr);
    } else if (std::strcmp(compile_spec.key, kImportForeverKey) == 0) {
      setting.mImportForever = *static_cast<char*>(compile_spec.value.buffer);
      LogInfo(
          "NeuronBackend",
          "IsImportForever Enable : %d",
          setting.mImportForever);
    } else if (std::strcmp(compile_spec.key, kSharedWeightsKey) == 0) {
      setting.mSharedWeights = true;
      std::string shared_weights_key(
          static_cast<char*>(compile_spec.value.buffer),
          compile_spec.value.nbytes);
      LogInfo(
          "NeuronBackend",
          "SharedWeights Enabled for %s",
          shared_weights_key.c_str());
      std::shared_ptr<NeuronSharedWeights> neuron_shared_weights;
      if (neuron_shared_weights_cache_.find(shared_weights_key) !=
          neuron_shared_weights_cache_.end()) {
        neuron_shared_weights =
            neuron_shared_weights_cache_.at(shared_weights_key).lock();
        if (neuron_shared_weights) {
          LogInfo(
              "NeuronBackend",
              "Reusing cached shared weights with key %s",
              shared_weights_key.c_str());
          delegate->SetSharedWeights(neuron_shared_weights);
          continue;
        } else {
          LogInfo(
              "NeuronBackend",
              "Shared weights cache expired: %s",
              shared_weights_key.c_str());
          neuron_shared_weights_cache_.erase(shared_weights_key); // Expired
        }
      }
      const NamedDataMap* named_data_map = context.get_named_data_map();
      Result<FreeableBuffer> shared_weights =
          named_data_map->get_data(shared_weights_key.c_str());

      if (shared_weights.ok()) {
        LogInfo(
            "NeuronBackend",
            "Loaded shared weights from named_data_map. Size: %zu",
            shared_weights.get().size());
        FreeableBuffer& buffer = shared_weights.get();
        neuron_shared_weights =
            std::make_shared<NeuronSharedWeights>(std::move(buffer));
        delegate->SetSharedWeights(neuron_shared_weights);
        neuron_shared_weights_cache_[shared_weights_key] =
            neuron_shared_weights;
      } else {
        LogError(
            "NeuronBackend",
            "Failed to load shared weights from named_data_map.");
        return Error::Internal;
      }
    } else {
      LogWarn("NeuronBackend", "unknown compile spec: %s", compile_spec.key);
    }
  }
  auto Payload = NeuronPayload(processed->data(), processed->size());

  LogInfo(
      "NeuronBackend",
      "version %u, input %u, output %u, length %u, payload size: %zu",
      Payload.Header.Version,
      Payload.Header.InputCount,
      Payload.Header.OutputCount,
      Payload.Header.DataLen,
      processed->size());

  int res = delegate->LoadCompiledNetwork(Payload, setting);
  return res == NEURON_NO_ERROR ? delegate : nullptr;
}

Error NeuronBackend::execute(
    ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  NeuronExecuTorchDelegate* delegate =
      reinterpret_cast<NeuronExecuTorchDelegate*>(handle);
  return delegate->execute(context, args);
}

void NeuronBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    NeuronExecuTorchDelegate* delegate =
        reinterpret_cast<NeuronExecuTorchDelegate*>(handle);
    delegate->~NeuronExecuTorchDelegate();
  }
}

bool NeuronBackend::is_available() const {
  return true;
}

Error NeuronExecuTorchDelegate::execute(
    BackendExecutionContext& context,
    Span<EValue*> args) const {
  if (HintNeuronBackend(args) != NEURON_NO_ERROR) {
    return Error::InvalidState;
  };

  ET_CHECK_OR_RETURN_ERROR(
      CheckDimOrder(args) == NEURON_NO_ERROR,
      Internal,
      "Expecting default dim_order but got a non default dim_order tensor input");

  PrepareInputsOuputs(args);

  auto allocator =
      dynamic_cast<neuron::BufferAllocator*>(context.get_temp_allocator());

  size_t inputCount = mInputSizes.size() + neuron_shared_weights_.size();
  size_t outputCount = mOutputSizes.size();

  for (size_t i = 0; i < inputCount; i++) {
    auto data_ptr = mPreparedInputs[i].data_ptr;
    auto data_size = mPreparedInputs[i].size;
    if (IsCached</*isInput=*/true>(i, data_ptr)) {
      continue;
    };
    auto unit = allocator != nullptr ? allocator->Find(data_ptr) : nullptr;
    if (unit) {
      UpdateCache<true>(i, data_ptr);
      size_t offset = (char*)data_ptr - (char*)unit->GetAddress();
      mExecutor.SetInputOutputFromMemory</*isInput*/ true>(
          i, unit->GetNeuronMemory(), offset, data_size);
    } else {
      mExecutor.SetInputOutput</*isInput=*/true>(i, data_ptr, data_size);
    }
  }

  for (size_t o = 0; o < outputCount; o++) {
    auto data_ptr = mPreparedOutputs[o].data_ptr;
    auto data_size = mPreparedOutputs[o].size;
    if (IsCached</*isInput=*/false>(o, data_ptr)) {
      continue;
    };
    auto unit = allocator != nullptr ? allocator->Find(data_ptr) : nullptr;
    if (unit) {
      UpdateCache</*isInput=*/false>(o, data_ptr);
      size_t offset = (char*)data_ptr - (char*)unit->GetAddress();
      mExecutor.SetInputOutputFromMemory</*isInput*/ false>(
          o, unit->GetNeuronMemory(), offset, data_size);
    } else {
      mExecutor.SetInputOutput</*isInput=*/false>(o, data_ptr, data_size);
    }
  }

  return mExecutor.Compute() == NEURON_NO_ERROR ? Error::Ok
                                                : Error::InvalidState;
};

int NeuronExecuTorchDelegate::HintNeuronBackend(Span<EValue*> args) const {
  auto HintImportForever = [this](Span<EValue*> args) -> int {
    auto& allocator = GET_NEURON_ALLOCATOR;
    size_t inputCount = mInputSizes.size(), outputCount = mOutputSizes.size();
    for (int i = 0; i < inputCount; i++) {
      auto data_ptr = args[i]->toTensor().data_ptr();
      if (mHasImported.count(data_ptr)) {
        continue;
      }
      auto unit = allocator.Find(data_ptr);
      if (unit) {
        mExecutor.SetInputOutputFromMemory</*isInput*/ true>(
            i, unit->GetNeuronMemory(), 0, unit->GetSize());
        mHasImported.insert(data_ptr);
      }
    }
    for (int o = inputCount; o < inputCount + outputCount; o++) {
      auto data_ptr = args[o]->toTensor().data_ptr();
      if (mHasImported.count(data_ptr)) {
        continue;
      }
      auto output_index = o - inputCount;
      auto unit = allocator.Find(data_ptr);
      if (unit) {
        mExecutor.SetInputOutputFromMemory</*isInput*/ false>(
            output_index, unit->GetNeuronMemory(), 0, unit->GetSize());
        mHasImported.insert(data_ptr);
      }
    }
    return NEURON_NO_ERROR;
  };
  if (mSettings.mImportForever) {
    CHECK_NO_ERROR(HintImportForever(args));
  }
  return NEURON_NO_ERROR;
}

} // namespace neuron
} // namespace backends
} // namespace executorch

namespace {
auto cls = executorch::backends::neuron::NeuronBackend();
executorch::runtime::Backend backend{"NeuropilotBackend", &cls};
static auto success_with_compiler =
    executorch::runtime::register_backend(backend);
} // namespace
