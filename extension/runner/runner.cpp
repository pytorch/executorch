/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner/runner.h>

namespace torch::executor {

Runner::Runner(
    std::unique_ptr<DataLoader> dataLoader,
    std::unique_ptr<MemoryAllocator> memoryAllocator)
    : dataLoader_(std::move(dataLoader)),
      memoryAllocator_(std::move(memoryAllocator)) {
  auto program = Program::load(dataLoader_.get());
  if (!program.ok()) {
    throw std::runtime_error(
        "Failed to load the program: " +
        std::to_string(error_code_t(program.error())));
  }
  program_ = std::make_unique<Program>(std::move(program.get()));
}

Error Runner::run(
    const std::string& methodName,
    const std::vector<EValue>& inputs,
    std::vector<EValue>& outputs) {
  auto error = loadMethod(methodName);
  if (error != Error::Ok) {
    return error;
  }
  auto& method = methods_.at(methodName).method;
  for (auto index = 0; index < inputs.size(); ++index) {
    error = method->set_input(inputs[index], index);
    if (error != Error::Ok) {
      return error;
    }
  }
  error = method->execute();
  if (error != Error::Ok) {
    return error;
  }
  const auto outputsSize = method->outputs_size();
  outputs.resize(outputsSize);

  return method->get_outputs(outputs.data(), outputsSize);
}

std::vector<std::string> Runner::methodNames() const {
  const auto methodCount = program_->num_methods();
  std::vector<std::string> result;
  result.reserve(methodCount);

  for (auto index = 0; index < methodCount; ++index) {
    result.emplace_back(program_->get_method_name(index).get());
  }
  return result;
}

Error Runner::loadMethod(const std::string& methodName) {
  if (!methods_.count(methodName)) {
    MethodHolder methodHolder;
    const auto methodMetadata = program_->method_meta(methodName.c_str());
    if (!methodMetadata.ok()) {
      return methodMetadata.error();
    }
    const auto plannedBuffersCount =
        methodMetadata->num_memory_planned_buffers();
    methodHolder.plannedBuffers.reserve(plannedBuffersCount);
    methodHolder.plannedSpans.reserve(plannedBuffersCount);

    for (auto index = 0; index < plannedBuffersCount; ++index) {
      const auto bufferSize =
          methodMetadata->memory_planned_buffer_size(index).get();
      methodHolder.plannedBuffers.emplace_back(bufferSize);
      methodHolder.plannedSpans.emplace_back(
          methodHolder.plannedBuffers.back().data(), bufferSize);
    }
    methodHolder.plannedMemory = std::make_unique<HierarchicalAllocator>(Span(
        methodHolder.plannedSpans.data(), methodHolder.plannedSpans.size()));
    methodHolder.memoryManager = std::make_unique<MemoryManager>(
        memoryAllocator_.get(), methodHolder.plannedMemory.get());

    auto method = program_->load_method(
        methodName.c_str(), methodHolder.memoryManager.get());
    if (!method.ok()) {
      return method.error();
    }
    methodHolder.method = std::make_unique<Method>(std::move(method.get()));
    methods_.emplace(methodName, std::move(methodHolder));
  }
  return Error::Ok;
}

} // namespace torch::executor
