/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include "MultiModelLoader.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::executor {

template <typename IdType>
void MultiModelLoader<IdType>::LoadModels() {
  // Init empty model instance map
  ET_LOG(Info, "cmodi LoadModels() 1");
  for (const auto& [id, _] : mModelPathMap) {
    ET_CHECK_MSG(
        !HasModel(id),
        "Model is already initialized before calling LoadModels.");
    ET_LOG(Info, "cmodi LoadModels() 2");
    mModelInstanceMap[id] = nullptr;
  }
  const size_t numModels = mModelPathMap.size();
  ET_LOG(Info, "cmodi LoadModels() 3");
  if (!AllowModelsCoexist()) {
    ET_LOG(Info, "cmodi LoadModels() 4");
    SelectModel(mDefaultModelId);
    ET_LOG(Info, "cmodi LoadModels() 5");
    ET_CHECK_MSG(
        GetModelInstance() == nullptr,
        "Model is already initialized before calling LoadModels.");
    ET_LOG(Info, "cmodi LoadModels() 6");
    void* instance = CreateModelInstance(mModelPathMap[mDefaultModelId]);
    ET_LOG(Info, "cmodi LoadModels() 7");
    SetModelInstance(instance);
    ET_LOG(Info, "cmodi LoadModels() 8");
    ET_LOG(
        Debug,
        "LoadModels(): Loaded single exclusive model (Total=%zu)",
        numModels);
    return;
  }
  for (const auto& [id, modelPath] : mModelPathMap) {
    ET_LOG(Info, "cmodi LoadModels() 9");
    SelectModel(id);
    ET_LOG(Info, "cmodi LoadModels() 10");
    ET_CHECK_MSG(
        GetModelInstance() == nullptr,
        "Model is already initialized before calling LoadModels.");
    ET_LOG(Info, "cmodi LoadModels() 11");
    void* instance = CreateModelInstance(modelPath);
    ET_LOG(Info, "cmodi LoadModels() 12");
    SetModelInstance(instance);
    ET_LOG(Info, "cmodi LoadModels() 13");
  }
  SelectModel(mDefaultModelId); // Select the default instance
  ET_LOG(Info, "cmodi LoadModels() 14");
  ET_LOG(Debug, "LoadModels(): Loaded multiple models (Total=%zu)", numModels);
}

template <typename IdType>
void MultiModelLoader<IdType>::ReleaseModels() {
  if (!AllowModelsCoexist()) {
    // Select the current instance
    ReleaseModelInstance(GetModelInstance());
    SetModelInstance(nullptr);
    return;
  }

  for (const auto& [id, _] : mModelInstanceMap) {
    SelectModel(id);
    ReleaseModelInstance(GetModelInstance());
    SetModelInstance(nullptr);
  }
}

template <typename IdType>
void* MultiModelLoader<IdType>::GetModelInstance() const {
  ET_DCHECK_MSG(
      HasModel(mCurrentModelId),
      "Invalid id: %s",
      GetIdString(mCurrentModelId).c_str());
  return mModelInstanceMap.at(mCurrentModelId);
}

template <typename IdType>
void MultiModelLoader<IdType>::SetModelInstance(void* instance) {
  ET_DCHECK_MSG(
      HasModel(mCurrentModelId),
      "Invalid id: %s",
      GetIdString(mCurrentModelId).c_str());
  mModelInstanceMap[mCurrentModelId] = instance;
}

template <typename IdType>
void MultiModelLoader<IdType>::SetDefaultModelId(const IdType& id) {
  mDefaultModelId = id;
}

template <typename IdType>
IdType MultiModelLoader<IdType>::GetModelId() const {
  return mCurrentModelId;
}

template <typename IdType>
void MultiModelLoader<IdType>::SelectModel(const IdType& id) {
  ET_CHECK_MSG(HasModel(id), "Invalid id: %s", GetIdString(id).c_str());

  if (mCurrentModelId == id) {
    return; // Do nothing
  } else if (AllowModelsCoexist()) {
    mCurrentModelId = id;
    return;
  }

  // Release current instance if already loaded
  if (HasModel(mCurrentModelId) && GetModelInstance() != nullptr) {
    ReleaseModelInstance(GetModelInstance());
    SetModelInstance(nullptr);
  }

  // Load new instance
  mCurrentModelId = id;
  void* newInstance = CreateModelInstance(mModelPathMap[id]);
  SetModelInstance(newInstance);
}

template <typename IdType>
size_t MultiModelLoader<IdType>::GetNumModels() const {
  ET_CHECK_MSG(
      mModelInstanceMap.size() == mModelPathMap.size(),
      "Please ensure that LoadModels() is called first.");
  return mModelInstanceMap.size();
}

template <typename IdType>
const std::string& MultiModelLoader<IdType>::GetModelPath() const {
  ET_CHECK_MSG(
      HasModel(mCurrentModelId),
      "Invalid id: %s",
      GetIdString(mCurrentModelId).c_str());
  return mModelPathMap.at(mCurrentModelId);
}

template <typename IdType>
void MultiModelLoader<IdType>::AddModel(
    const IdType& id,
    const std::string& modelPath) {
  if (HasModel(id)) {
    ET_LOG(
        Info,
        "Overlapping model identifier detected. Replacing existing model instance.");
    auto& oldInstance = mModelInstanceMap[id];
    if (oldInstance != nullptr)
      ReleaseModelInstance(oldInstance);
    oldInstance = nullptr;
  }
  mModelPathMap[id] = modelPath;

  // Create runtime immediately if can coexist
  mModelInstanceMap[id] = AllowModelsCoexist()
      ? CreateModelInstance(mModelPathMap[mDefaultModelId])
      : nullptr;
}

template <typename IdType>
bool MultiModelLoader<IdType>::HasModel(const IdType& id) const {
  return mModelInstanceMap.find(id) != mModelInstanceMap.end();
}

template <typename IdType>
std::string MultiModelLoader<IdType>::GetIdString(const IdType& id) {
  std::ostringstream ss;
  ss << id;
  return ss.str();
}

// Explicit instantiation of MultiModelLoader for some integral Id types
template class MultiModelLoader<int>;
template class MultiModelLoader<size_t>;

} // namespace torch::executor
