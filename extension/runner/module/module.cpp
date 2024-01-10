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
    const bool useMlock,
    const bool ignoreMlockErrors)
    : filePath_(filePath),
      useMlock_(useMlock),
      ignoreMlockErrors_(ignoreMlockErrors) {
  ET_CHECK(!filePath_.empty());
}

Error Module::load() {
  if (runner_) {
    return Error::Ok;
  }
  const auto mlockConfig = useMlock_
      ? ignoreMlockErrors_
          ? util::MmapDataLoader::MlockConfig::UseMlockIgnoreErrors
          : util::MmapDataLoader::MlockConfig::UseMlock
      : util::MmapDataLoader::MlockConfig::NoMlock;
  auto dataLoader = util::MmapDataLoader::from(filePath_.c_str(), mlockConfig);
  if (!dataLoader.ok()) {
    return dataLoader.error();
  }
  runner_ = std::make_unique<Runner>(
      std::make_unique<util::MmapDataLoader>(std::move(dataLoader.get())),
      std::make_unique<util::MallocMemoryAllocator>());
  const auto status = runner_->load();
  if (status != Error::Ok) {
    runner_.reset();
  }
  return status;
}

Error Module::forward(
    const std::vector<EValue>& inputs,
    std::vector<EValue>& outputs) {
  if (!runner_) {
    const auto status = load();
    if (status != Error::Ok) {
      return status;
    }
  }
  return runner_->run("forward", inputs, outputs);
}

std::vector<std::string> Module::methodNames() const {
  if (!runner_) {
    return {};
  }
  return runner_->methodNames();
}

} // namespace torch::executor
