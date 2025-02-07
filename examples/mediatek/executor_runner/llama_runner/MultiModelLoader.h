/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace example {

template <typename IdType = size_t>
class MultiModelLoader {
 public:
  using ModelPathMap = std::unordered_map<IdType, std::string>;
  using ModelInstanceMap = std::unordered_map<IdType, void*>;

  explicit MultiModelLoader(
      const ModelPathMap& modelPathMap,
      const IdType defaultModelId = {})
      : mModelPathMap(modelPathMap),
        mDefaultModelId(defaultModelId),
        mCurrentModelId(defaultModelId) {}

  explicit MultiModelLoader(
      const std::string& modelPath,
      const IdType defaultModelId = {})
      : mModelPathMap({{defaultModelId, modelPath}}),
        mDefaultModelId(defaultModelId),
        mCurrentModelId(defaultModelId) {}

  virtual ~MultiModelLoader() {}

 protected:
  // Initialize all models if they can coexist, otherwise initialize the default
  // model.
  void LoadModels();

  // Release all active model instances.
  void ReleaseModels();

  // Get the current model instance.
  void* GetModelInstance() const;

  // Set the current model instance.
  void SetModelInstance(void* modelInstance);

  // Set the default active model after LoadModels() has been called.
  void SetDefaultModelId(const IdType& id);

  // Get the id of the current model instance.
  IdType GetModelId() const;

  // Select the model of given id to be active.
  void SelectModel(const IdType& id);

  // Get total number of models.
  size_t GetNumModels() const;

  // Get the model path of the current active model.
  const std::string& GetModelPath() const;

  // Add new model post initialization, and returns the model id.
  void AddModel(const IdType& id, const std::string& modelPath);

  bool HasModel(const IdType& id) const;

  static std::string GetIdString(const IdType& id);

 private:
  // Create and returns a model instance given a model path. To be implemented
  // by subclass.
  virtual void* CreateModelInstance(const std::string& modelPath) = 0;

  // Release a model instance. To be implemented by subclass.
  virtual void ReleaseModelInstance(void* modelInstance) = 0;

  // Determine whether multiple models are allowed to be alive concurrently.
  virtual bool AllowModelsCoexist() const {
    return false;
  }

 private:
  ModelPathMap mModelPathMap;
  ModelInstanceMap mModelInstanceMap;
  IdType mDefaultModelId = 0;
  IdType mCurrentModelId = 0;
};

} // namespace example
