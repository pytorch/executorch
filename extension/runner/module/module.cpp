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

Module::Module(const std::string& filePath)
    : Runner(
          ({
            auto dataLoader = util::MmapDataLoader::from(
                filePath.c_str(),
                util::MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
            if (!dataLoader.ok()) {
              throw std::runtime_error("Failed to load file: " + filePath);
            }
            std::make_unique<util::MmapDataLoader>(std::move(dataLoader.get()));
          }),
          std::make_unique<util::MallocMemoryAllocator>()) {}

} // namespace torch::executor
