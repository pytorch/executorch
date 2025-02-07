/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <executorch/runtime/executor/method.h>

#include "MultiModelLoader.h"

namespace example {

struct BufferInfo {
  void* data = nullptr;
  size_t nbytes = 0;
  size_t nbytesUsed = 0;
};

using MultiTokenSizeModelLoader = MultiModelLoader<size_t>;
using ModelPathMap = MultiTokenSizeModelLoader::ModelPathMap;

class ModelChunk : protected MultiTokenSizeModelLoader {
 public:
  explicit ModelChunk(
      const ModelPathMap& modelPathMap,
      const size_t initBatchSize = 1)
      : MultiTokenSizeModelLoader(modelPathMap, initBatchSize),
        mTokenBatchSize(initBatchSize) {}

  explicit ModelChunk(
      const std::string& modelPath,
      const size_t initBatchSize = 1)
      : MultiTokenSizeModelLoader(modelPath, initBatchSize),
        mTokenBatchSize(initBatchSize) {}

  ~ModelChunk() {}

  virtual void Initialize();

  virtual void Release();

  virtual void Run();

  virtual bool HotSwapModel(const size_t tokenBatchSize);

  void
  SetInputBuffer(const void* data, const size_t size, const size_t index = 0);

  void SetInputBuffer(const BufferInfo& bufferInfo, const size_t index = 0);

  BufferInfo GetInputBuffer(const size_t index = 0);

  BufferInfo GetOutputBuffer(const size_t index = 0);

  void LogIoSummary();

 protected:
  // Check if model chunk has been initialized
  bool Initialized();

  // Get model IO info after model has been loaded
  void GetModelIoInfo();

  // Update IO sizes actually used by the model
  void UpdateModelIoInfo();

  // Model IO linkage to share the same buffer among a pair of linked input and
  // output
  void LinkModelIO(const size_t inputIndex, const size_t outputIndex);

  // Return the input index that the given output should share the same buffer
  std::optional<size_t> GetLinkedInputIndex(const size_t outputIndex) const;

  // Assign input buffers to model inputs using backend APIs
  void SetBackendInputs();

  // Assign output buffers to model outputs using backend APIs
  void SetBackendOutputs();

  // Allocate buffers for model IOs
  void AllocateIoBuffers();

  // Release allocated buffers for model IOs
  void ReleaseIoBuffers();

  executorch::runtime::Method& GetModelMethod();

 private:
  // Override the virtual functions
  void* CreateModelInstance(const std::string& modelPath) override;

  void ReleaseModelInstance(void* modelInstance) override;

 private:
  bool AllowModelsCoexist() const override {
    return false;
  }

 protected:
  // State of initialization
  bool mIsInitialized = false;

  // The number of input tokens the the fixed-shape model takes
  size_t mTokenBatchSize = 1;

  // Input/Output buffers info
  std::vector<BufferInfo> mInputBufferInfos;
  std::vector<BufferInfo> mOutputBufferInfos;

  // Model IO linkage, where linked IO will share the same buffer
  std::unordered_map<size_t, size_t> mModelOutToInIndexLinks;
};

} // namespace example
