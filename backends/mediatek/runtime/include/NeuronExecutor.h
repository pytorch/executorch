/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include "NeuronLog.h"
#include "api/NeuronAdapter.h"
#include "api/NeuronAdapterShim.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace executor {
namespace neuron {

struct NeuronDeleter {
  void operator()(NeuronModel* model) {
    if (model != nullptr) {
      NeuronModel_free(model);
    }
  }
  void operator()(NeuronCompilation* compilation) {
    if (compilation != nullptr) {
      NeuronCompilation_free(compilation);
    }
  }
  void operator()(NeuronExecution* execution) {
    if (execution != nullptr) {
      NeuronExecution_free(execution);
    }
  }
  void operator()(NeuronMemory* memory) {
    if (memory != nullptr) {
      NeuronMemory_free(memory);
    }
  }
};

class NeuronExecutor {
 public:
  explicit NeuronExecutor();

  int LoadFromCompiledNetwork(
      const void* buffer,
      size_t size,
      int inputCount,
      int outputCount,
      std::string& runtimeOption);

  template <bool isInput>
  int SetInputOutput(uint32_t index, void* buffer, size_t length) const {
    CHECK_VALID_PTR(buffer);
    CHECK_VALID_PTR(mExecution);
    return isInput ? NeuronExecution_setInput(
                         mExecution.get(), index, nullptr, buffer, length)
                   : NeuronExecution_setOutput(
                         mExecution.get(), index, nullptr, buffer, length);
  }

  template <bool isInput>
  int SetInputOutputFromMemory(
      uint32_t index,
      const NeuronMemory* memory,
      size_t offset,
      size_t length) const {
    CHECK_VALID_PTR(memory);
    CHECK_VALID_PTR(mExecution);
    return isInput
        ? NeuronExecution_setInputFromMemory(
              mExecution.get(), index, nullptr, memory, offset, length)
        : NeuronExecution_setOutputFromMemory(
              mExecution.get(), index, nullptr, memory, offset, length);
  }

  template <bool isInput>
  size_t GetInputOutputPaddedSize(int32_t index) const {
    CHECK_VALID_PTR(mCompilation);
    size_t size = 0;
    auto res = isInput
        ? NeuronCompilation_getInputPaddedSize(mCompilation.get(), index, &size)
        : NeuronCompilation_getOutputPaddedSize(
              mCompilation.get(), index, &size);
    return res == NEURON_NO_ERROR ? size : 0;
  }

  int Compute() const {
    CHECK_VALID_PTR(mExecution);
    return NeuronExecution_compute(mExecution.get());
  }

  bool IsValid() const {
    return mExecution != nullptr;
  }

 private:
  std::unique_ptr<NeuronModel, NeuronDeleter> mModel;

  std::unique_ptr<NeuronCompilation, NeuronDeleter> mCompilation;

  std::unique_ptr<NeuronExecution, NeuronDeleter> mExecution;

  std::vector<size_t> mInputSizes;

  std::vector<size_t> mOutputSizes;

 private:
  NeuronExecutor(const NeuronExecutor&);

  NeuronExecutor operator=(const NeuronExecutor&);
};

} // namespace neuron
} // namespace executor
} // namespace torch
