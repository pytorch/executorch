/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

/**
 * Unwrap a Result to obtain its value (direct object, not a pointer).
 * If the Result contains an error, propagate the error via trivial function
 * return. The macro wraps the object into a unique_ptr.
 *
 * Note: A function using ET_UNWRAP_UNIQUE should itself return a Result or
 * Error.
 *
 * @param[in] result__ Expression yielding the result to unwrap.
 */
#define ET_UNWRAP_UNIQUE(result__)                                     \
  ({                                                                   \
    auto et_result__ = (result__);                                     \
    if (!et_result__.ok()) {                                           \
      return et_result__.error();                                      \
    }                                                                  \
    std::make_unique<std::remove_reference_t<decltype(*et_result__)>>( \
        std::move(*et_result__));                                      \
  })

namespace torch::executor {

Module::Module(
    const std::string& filePath,
    const Module::MlockConfig mlockConfig)
    : filePath_(filePath),
      mlockConfig_(mlockConfig),
      memoryAllocator_(std::make_unique<util::MallocMemoryAllocator>()) {
  runtime_init();
}

Module::Module(
    std::unique_ptr<DataLoader> dataLoader,
    std::unique_ptr<MemoryAllocator> memoryAllocator,
    std::unique_ptr<EventTracer> eventTracer)
    : dataLoader_(std::move(dataLoader)),
      memoryAllocator_(
          std::move(memoryAllocator)
              ?: std::make_unique<util::MallocMemoryAllocator>()),
      eventTracer_(std::move(eventTracer)) {
  runtime_init();
}

Error Module::load(const Program::Verification verification) {
  if (!isLoaded()) {
    if (!dataLoader_) {
      dataLoader_ = ET_UNWRAP_UNIQUE(
          util::MmapDataLoader::from(filePath_.c_str(), [this] {
            switch (mlockConfig_) {
              case MlockConfig::NoMlock:
                return util::MmapDataLoader::MlockConfig::NoMlock;
              case MlockConfig::UseMlock:
                return util::MmapDataLoader::MlockConfig::UseMlock;
              case MlockConfig::UseMlockIgnoreErrors:
                return util::MmapDataLoader::MlockConfig::UseMlockIgnoreErrors;
            }
            ET_ASSERT_UNREACHABLE();
          }()));
    };
    program_ = ET_UNWRAP_UNIQUE(Program::load(dataLoader_.get(), verification));
  }
  return Error::Ok;
}

bool Module::isLoaded() const {
  return program_ != nullptr;
}

Result<std::vector<std::string>> Module::methodNames() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  const auto methodCount = program_->num_methods();
  std::vector<std::string> result;
  result.reserve(methodCount);

  for (auto index = 0; index < methodCount; ++index) {
    result.emplace_back(program_->get_method_name(index).get());
  }
  return result;
}

Error Module::loadMethod(const std::string& methodName) {
  if (!isMethodLoaded(methodName)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());

    MethodHolder methodHolder;
    const auto methodMetadata =
        ET_UNWRAP(program_->method_meta(methodName.c_str()));
    const auto plannedBuffersCount =
        methodMetadata.num_memory_planned_buffers();
    methodHolder.plannedBuffers.reserve(plannedBuffersCount);
    methodHolder.plannedSpans.reserve(plannedBuffersCount);

    for (auto index = 0; index < plannedBuffersCount; ++index) {
      const auto bufferSize =
          methodMetadata.memory_planned_buffer_size(index).get();
      methodHolder.plannedBuffers.emplace_back(bufferSize);
      methodHolder.plannedSpans.emplace_back(
          methodHolder.plannedBuffers.back().data(), bufferSize);
    }
    methodHolder.plannedMemory = std::make_unique<HierarchicalAllocator>(Span(
        methodHolder.plannedSpans.data(), methodHolder.plannedSpans.size()));
    methodHolder.memoryManager = std::make_unique<MemoryManager>(
        memoryAllocator_.get(), methodHolder.plannedMemory.get());
    methodHolder.method = ET_UNWRAP_UNIQUE(program_->load_method(
        methodName.c_str(),
        methodHolder.memoryManager.get(),
        eventTracer_.get()));
    methods_.emplace(methodName, std::move(methodHolder));
  }
  return Error::Ok;
}

bool Module::isMethodLoaded(const std::string& methodName) const {
  return methods_.count(methodName);
}

Result<MethodMeta> Module::methodMeta(const std::string& methodName) {
  ET_CHECK_OK_OR_RETURN_ERROR(loadMethod(methodName));
  return methods_.at(methodName).method->method_meta();
}

Result<std::vector<EValue>> Module::execute(
    const std::string& methodName,
    const std::vector<EValue>& input) {
  ET_CHECK_OK_OR_RETURN_ERROR(loadMethod(methodName));
  auto& method = methods_.at(methodName).method;

  for (auto index = 0; index < input.size(); ++index) {
    ET_CHECK_OK_OR_RETURN_ERROR(method->set_input(input[index], index));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(method->execute());

  const auto outputsSize = method->outputs_size();
  std::vector<EValue> outputs(outputsSize);
  ET_CHECK_OK_OR_RETURN_ERROR(method->get_outputs(outputs.data(), outputsSize));

  return outputs;
}

} // namespace torch::executor
