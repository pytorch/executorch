/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner/module/module.h>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>

namespace torch::executor {

Module::Module(
    const std::string& filePath,
    const Module::MlockConfig mlockConfig)
    : Runner(nullptr, std::make_unique<util::MallocMemoryAllocator>()),
      filePath_(filePath),
      mlockConfig_(mlockConfig) {}

Error Module::load() {
  if (!isLoaded()) {
    auto dataLoader = util::MmapDataLoader::from(filePath_.c_str(), [this] {
      switch (mlockConfig_) {
        case MlockConfig::NoMlock:
          return util::MmapDataLoader::MlockConfig::NoMlock;
        case MlockConfig::UseMlock:
          return util::MmapDataLoader::MlockConfig::UseMlock;
        case MlockConfig::UseMlockIgnoreErrors:
          return util::MmapDataLoader::MlockConfig::UseMlockIgnoreErrors;
      }
    }());
    if (!dataLoader.ok()) {
      return dataLoader.error();
    }
    dataLoader_ =
        std::make_unique<util::MmapDataLoader>(std::move(dataLoader.get()));
  }
  return Runner::load();
}

} // namespace torch::executor
