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
  auto& method = methods_.at(methodName).method_;
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

Error Runner::loadMethod(const std::string& methodName) {
  if (!methods_.count(methodName)) {
    auto methodMetadata = program_->method_meta(methodName.c_str());
    if (!methodMetadata.ok()) {
      return methodMetadata.error();
    }
    std::vector<size_t> bufferSizes;
    const auto plannedBuffersCount =
        methodMetadata->num_memory_planned_buffers();
    for (auto id = 0; id < plannedBuffersCount; ++id) {
      const size_t bufferSize =
          methodMetadata->memory_planned_buffer_size(id).get();
      bufferSizes.emplace_back(bufferSize);
    }
    auto memory = std::make_unique<Memory>(bufferSizes, memoryAllocator_);

    auto method = program_->load_method(
        methodName.c_str(), memory->getMemoryManager().get());
    if (!method.ok()) {
      return method.error();
    }
    methods_[methodName] = {
        std::move(memory), std::make_unique<Method>(std::move(method.get()))};
  }
  return Error::Ok;
}

} // namespace torch::executor
