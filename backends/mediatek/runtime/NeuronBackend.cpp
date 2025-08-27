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

#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/exec_aten/util/dim_order_util.h"

#include <algorithm>
#include <memory>
#include <new>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace neuron {

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

Result<DelegateHandle*> NeuronBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  NeuronDelegateSetting setting;
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

  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  NeuronExecuTorchDelegate* delegate =
      runtime_allocator->allocateInstance<NeuronExecuTorchDelegate>();
  if (delegate == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  new (delegate) NeuronExecuTorchDelegate();

  if (delegate == nullptr) {
    return nullptr;
  }
  auto res = delegate->LoadCompiledNetwork(Payload, setting);
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

  auto allocator =
      dynamic_cast<neuron::BufferAllocator*>(context.get_temp_allocator());
  size_t inputCount = mInputSizes.size(), outputCount = mOutputSizes.size();

  for (int i = 0; i < inputCount; i++) {
    auto tensor_in = args[i]->toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        runtime::is_contiguous_dim_order(
            tensor_in.dim_order().data(), tensor_in.dim()),
        Internal,
        "Expecting default dim_order but got a non default dim_order tensor for external input %u",
        i);

    auto data_ptr = args[i]->toTensor().data_ptr();
    auto data_size = args[i]->toTensor().nbytes();
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

  for (int o = inputCount; o < inputCount + outputCount; o++) {
    auto data_ptr = args[o]->toTensor().data_ptr();
    auto data_size = args[o]->toTensor().nbytes();
    auto output_index = o - inputCount;
    if (IsCached</*isInput=*/false>(output_index, data_ptr)) {
      continue;
    };
    auto unit = allocator != nullptr ? allocator->Find(data_ptr) : nullptr;
    if (unit) {
      UpdateCache</*isInput=*/false>(output_index, data_ptr);
      size_t offset = (char*)data_ptr - (char*)unit->GetAddress();
      mExecutor.SetInputOutputFromMemory</*isInput*/ false>(
          output_index, unit->GetNeuronMemory(), offset, data_size);
    } else {
      mExecutor.SetInputOutput</*isInput=*/false>(
          output_index, data_ptr, data_size);
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
