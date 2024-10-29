/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <executorch/extension/data_loader/file_data_loader.h>
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

namespace executorch {
namespace extension {

Module::Module(
    const std::string& file_path,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer)
    : file_path_(file_path),
      load_mode_(load_mode),
      memory_allocator_(std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  runtime::runtime_init();
}

Module::Module(
    std::unique_ptr<runtime::DataLoader> data_loader,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer)
    : data_loader_(std::move(data_loader)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  runtime::runtime_init();
}

Module::Module(
    std::shared_ptr<runtime::Program> program,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer)
    : program_(std::move(program)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  runtime::runtime_init();
}

runtime::Error Module::load(const runtime::Program::Verification verification) {
  if (!is_loaded()) {
    if (!data_loader_) {
      switch (load_mode_) {
        case LoadMode::File:
          data_loader_ =
              ET_UNWRAP_UNIQUE(FileDataLoader::from(file_path_.c_str()));
          break;
        case LoadMode::Mmap:
          data_loader_ = ET_UNWRAP_UNIQUE(MmapDataLoader::from(
              file_path_.c_str(), MmapDataLoader::MlockConfig::NoMlock));
          break;
        case LoadMode::MmapUseMlock:
          data_loader_ =
              ET_UNWRAP_UNIQUE(MmapDataLoader::from(file_path_.c_str()));
          break;
        case LoadMode::MmapUseMlockIgnoreErrors:
          data_loader_ = ET_UNWRAP_UNIQUE(MmapDataLoader::from(
              file_path_.c_str(),
              MmapDataLoader::MlockConfig::UseMlockIgnoreErrors));
          break;
      }
    };
    auto program = ET_UNWRAP_UNIQUE(
        runtime::Program::load(data_loader_.get(), verification));
    program_ = std::shared_ptr<runtime::Program>(
        program.release(), [](runtime::Program* pointer) { delete pointer; });
  }
  return runtime::Error::Ok;
}

runtime::Result<std::unordered_set<std::string>> Module::method_names() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  const auto method_count = program_->num_methods();
  std::unordered_set<std::string> result;
  result.reserve(method_count);

  for (auto index = 0; index < method_count; ++index) {
    result.emplace(program_->get_method_name(index).get());
  }
  return result;
}

runtime::Error Module::load_method(
    const std::string& method_name,
    torch::executor::EventTracer* event_tracer) {
  if (!is_method_loaded(method_name)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());

    MethodHolder method_holder;
    const auto method_metadata =
        ET_UNWRAP(program_->method_meta(method_name.c_str()));
    const auto planned_buffersCount =
        method_metadata.num_memory_planned_buffers();
    method_holder.planned_buffers.reserve(planned_buffersCount);
    method_holder.planned_spans.reserve(planned_buffersCount);

    for (auto index = 0; index < planned_buffersCount; ++index) {
      const auto buffer_size =
          method_metadata.memory_planned_buffer_size(index).get();
      method_holder.planned_buffers.emplace_back(buffer_size);
      method_holder.planned_spans.emplace_back(
          method_holder.planned_buffers.back().data(), buffer_size);
    }
    method_holder.planned_memory =
        std::make_unique<runtime::HierarchicalAllocator>(runtime::Span(
            method_holder.planned_spans.data(),
            method_holder.planned_spans.size()));
    method_holder.memory_manager = std::make_unique<runtime::MemoryManager>(
        memory_allocator_.get(),
        method_holder.planned_memory.get(),
        temp_allocator_.get());
    method_holder.method = ET_UNWRAP_UNIQUE(program_->load_method(
        method_name.c_str(),
        method_holder.memory_manager.get(),
        event_tracer ? event_tracer : this->event_tracer()));
    method_holder.inputs.resize(method_holder.method->inputs_size());
    methods_.emplace(method_name, std::move(method_holder));
  }
  return runtime::Error::Ok;
}

runtime::Result<runtime::MethodMeta> Module::method_meta(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  return methods_.at(method_name).method->method_meta();
}

runtime::Result<std::vector<runtime::EValue>> Module::execute(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  auto& inputs = methods_.at(method_name).inputs;

  for (size_t i = 0; i < input_values.size(); ++i) {
    if (!input_values[i].isNone()) {
      inputs[i] = input_values[i];
    }
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    ET_CHECK_OR_RETURN_ERROR(
        !inputs[i].isNone(), InvalidArgument, "input %zu is none", i);
  }
  ET_CHECK_OK_OR_RETURN_ERROR(method->set_inputs(
      exec_aten::ArrayRef<runtime::EValue>(inputs.data(), inputs.size())));
  ET_CHECK_OK_OR_RETURN_ERROR(method->execute());

  const auto outputs_size = method->outputs_size();
  std::vector<runtime::EValue> outputs(outputs_size);
  ET_CHECK_OK_OR_RETURN_ERROR(
      method->get_outputs(outputs.data(), outputs_size));

  return outputs;
}

runtime::Error Module::set_input(
    const std::string& method_name,
    const runtime::EValue& input_value,
    size_t input_index) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  methods_.at(method_name).inputs.at(input_index) = input_value;
  return runtime::Error::Ok;
}

runtime::Error Module::set_inputs(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& inputs = methods_.at(method_name).inputs;
  ET_CHECK_OR_RETURN_ERROR(
      inputs.size() == input_values.size(),
      InvalidArgument,
      "input size: %zu does not match method input size: %zu",
      input_values.size(),
      inputs.size());
  inputs = input_values;
  return runtime::Error::Ok;
}

runtime::Error Module::set_output(
    const std::string& method_name,
    runtime::EValue output_value,
    size_t output_index) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  ET_CHECK_OR_RETURN_ERROR(
      output_value.isTensor(),
      InvalidArgument,
      "output type: %zu is not tensor",
      (size_t)output_value.tag);
  const auto& output_tensor = output_value.toTensor();
  return method->set_output_data_ptr(
      output_tensor.mutable_data_ptr(), output_tensor.nbytes(), output_index);
}

} // namespace extension
} // namespace executorch
