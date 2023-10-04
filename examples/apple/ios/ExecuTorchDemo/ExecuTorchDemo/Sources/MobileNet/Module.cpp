/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Module.h"

#include <mutex>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

namespace torch::executor::demo {

Module::Module(const std::string& filePath) {
  static std::once_flag flag;
  std::call_once(flag, []() { runtime_init(); });
  auto dataLoader = util::MmapDataLoader::from(filePath.c_str());
  if (!dataLoader.ok()) {
    throw std::runtime_error("Failed to load file: " + filePath);
  }
  dataLoader_ =
      std::make_unique<util::MmapDataLoader>(std::move(dataLoader.get()));
  auto program = Program::load(dataLoader_.get());
  if (!program.ok()) {
    throw std::runtime_error(
        "Failed to load the program from file: " + filePath);
  }
  program_ = std::make_unique<Program>(std::move(program.get()));
  auto methodMetadata = program->method_meta("forward");
  if (!methodMetadata.ok()) {
    throw std::runtime_error("Failed to load the forward method metadata");
  }
  methodAllocator_ = std::make_unique<util::MallocMemoryAllocator>();
  const auto plannedBuffersCount = methodMetadata->num_memory_planned_buffers();
  for (auto id = 0; id < plannedBuffersCount; ++id) {
    const auto bufferSize = static_cast<size_t>(
        methodMetadata->memory_planned_buffer_size(id).get());
    plannedBuffers_.emplace_back(std::vector<uint8_t>(bufferSize));
    plannedSpans_.emplace_back(plannedBuffers_.back().data(), bufferSize);
  }
  plannedMemory_ = std::make_unique<HierarchicalAllocator>(
      Span(plannedSpans_.data(), plannedSpans_.size()));
  memoryManager_ = std::make_unique<MemoryManager>(
      methodAllocator_.get(), plannedMemory_.get());

  auto forward = program_->load_method("forward", memoryManager_.get());
  if (!forward.ok()) {
    throw std::runtime_error("Failed to load the forward method");
  }
  method_ = std::make_unique<Method>(std::move(forward.get()));
}

Error Module::forward(
    const std::vector<EValue>& inputs,
    std::vector<EValue>& outputs) {
  for (auto index = 0; index < inputs.size(); ++index) {
    Error error = method_->set_input(inputs[index], index);
    if (error != Error::Ok) {
      return error;
    }
  }
  Error error = method_->execute();
  if (error != Error::Ok) {
    return error;
  }
  const auto outputsSize = method_->outputs_size();
  outputs.resize(outputsSize);

  return method_->get_outputs(outputs.data(), outputsSize);
}

} // namespace torch::executor::demo
