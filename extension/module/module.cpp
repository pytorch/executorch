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
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
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
namespace ET_MODULE_NAMESPACE {

using ET_RUNTIME_NAMESPACE::MethodMeta;
using ET_RUNTIME_NAMESPACE::Program;

namespace {
runtime::Result<std::unique_ptr<runtime::DataLoader>> make_data_loader(
    const std::string& file_path,
    Module::LoadMode mode) {
  std::unique_ptr<runtime::DataLoader> data_loader;
  switch (mode) {
    case Module::LoadMode::File:
      data_loader = ET_UNWRAP_UNIQUE(FileDataLoader::from(file_path.c_str()));
      break;
    case Module::LoadMode::Mmap:
      data_loader = ET_UNWRAP_UNIQUE(MmapDataLoader::from(
          file_path.c_str(), MmapDataLoader::MlockConfig::NoMlock));
      break;
    case Module::LoadMode::MmapUseMlock:
      data_loader = ET_UNWRAP_UNIQUE(MmapDataLoader::from(file_path.c_str()));
      break;
    case Module::LoadMode::MmapUseMlockIgnoreErrors:
      data_loader = ET_UNWRAP_UNIQUE(MmapDataLoader::from(
          file_path.c_str(),
          MmapDataLoader::MlockConfig::UseMlockIgnoreErrors));
      break;
  }
  return data_loader;
}
} // namespace

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
    const std::string& file_path,
    const std::string& data_map_path,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer)
    : file_path_(file_path),
      load_mode_(load_mode),
      memory_allocator_(std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  if (!data_map_path.empty()) {
    data_files_.push_back(data_map_path);
  }
  runtime::runtime_init();
}

Module::Module(
    const std::string& file_path,
    std::vector<std::string> data_files,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer)
    : file_path_(file_path),
      data_files_(std::move(data_files)),
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
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader)
    : data_loader_(std::move(data_loader)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  if (data_map_loader) {
    data_map_loaders_.push_back(std::move(data_map_loader));
  }
  runtime::runtime_init();
}

Module::Module(
    std::shared_ptr<Program> program,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader)
    : program_(std::move(program)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  if (data_map_loader) {
    data_map_loaders_.push_back(std::move(data_map_loader));
  }
  runtime::runtime_init();
}

runtime::Error Module::load(const Program::Verification verification) {
  if (!is_loaded()) {
    if (!data_loader_) {
      data_loader_ = ET_UNWRAP(make_data_loader(file_path_, load_mode_));
    }
    if (data_files_.size() > 0) {
      ET_CHECK_OR_RETURN_ERROR(
          data_files_.size() == 1,
          NotImplemented,
          "Multiple named data map paths are not supported yet.");
      for (const auto& data_file : data_files_) {
        data_map_loaders_.push_back(
            ET_UNWRAP(make_data_loader(data_file, load_mode_)));
      }
    }

    if (data_map_loaders_.size() > 0) {
      ET_CHECK_OR_RETURN_ERROR(
          data_map_loaders_.size() == 1 && merged_data_map_ == nullptr,
          NotImplemented,
          "Multiple named data map loaders are not supported yet.");
      // TODO(lfq): support multiple named data map loaders.
      merged_data_map_ =
          ET_UNWRAP_UNIQUE(FlatTensorDataMap::load(data_map_loaders_[0].get()));
    }

    auto program =
        ET_UNWRAP_UNIQUE(Program::load(data_loader_.get(), verification));
    program_ = std::shared_ptr<Program>(
        program.release(), [](Program* pointer) { delete pointer; });
  }
  return runtime::Error::Ok;
}

runtime::Result<size_t> Module::num_methods() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  return program_->num_methods();
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
    runtime::HierarchicalAllocator* planned_memory,
    torch::executor::EventTracer* event_tracer) {
  if (!is_method_loaded(method_name)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());

    MethodHolder method_holder;

    if (!planned_memory) {
      const auto method_metadata =
          ET_UNWRAP(program_->method_meta(method_name.c_str()));
      const auto planned_buffers_count =
          method_metadata.num_memory_planned_buffers();
      method_holder.planned_buffers.reserve(planned_buffers_count);
      method_holder.planned_spans.reserve(planned_buffers_count);

      for (auto index = 0; index < planned_buffers_count; ++index) {
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
      planned_memory = method_holder.planned_memory.get();
    }
    method_holder.memory_manager = std::make_unique<runtime::MemoryManager>(
        memory_allocator_.get(), planned_memory, temp_allocator_.get());
    method_holder.method = ET_UNWRAP_UNIQUE(program_->load_method(
        method_name.c_str(),
        method_holder.memory_manager.get(),
        event_tracer ? event_tracer : this->event_tracer(),
        merged_data_map_.get()));
    methods_.emplace(method_name, std::move(method_holder));
  }
  return runtime::Error::Ok;
}

ET_NODISCARD runtime::Result<Method*> Module::method(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  return methods_[method_name].method.get();
}

runtime::Result<MethodMeta> Module::method_meta(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  return program_->method_meta(method_name.c_str());
}

runtime::Result<std::vector<runtime::EValue>> Module::execute(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  for (auto index = 0; index < input_values.size(); ++index) {
    ET_CHECK_OK_OR_RETURN_ERROR(method->set_input(input_values[index], index));
  }
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
  auto& method = methods_.at(method_name).method;
  return method->set_input(input_value, input_index);
}

runtime::Error Module::set_inputs(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  return method->set_inputs(executorch::aten::ArrayRef<runtime::EValue>(
      input_values.data(), input_values.size()));
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

runtime::Error Module::set_outputs(
    const std::string& method_name,
    const std::vector<runtime::EValue>& output_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  const auto outputs_size = method->outputs_size();
  ET_CHECK_OR_RETURN_ERROR(
      output_values.size() == outputs_size,
      InvalidArgument,
      "output size: %zu is not equal to method output size: %zu",
      output_values.size(),
      outputs_size);
  for (auto index = 0; index < outputs_size; ++index) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        set_output(method_name, output_values[index], index));
  }
  return runtime::Error::Ok;
}

runtime::Result<std::vector<runtime::EValue>> Module::get_outputs(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  const auto outputs_size = method->outputs_size();
  std::vector<runtime::EValue> outputs(outputs_size);
  ET_CHECK_OK_OR_RETURN_ERROR(
      method->get_outputs(outputs.data(), outputs_size));
  return outputs;
}

runtime::Result<runtime::EValue> Module::get_output(
    const std::string& method_name,
    size_t output_index) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  ET_CHECK_OR_RETURN_ERROR(
      output_index < method->outputs_size(),
      InvalidArgument,
      "output index: %zu is out of range",
      output_index);
  return method->get_output(output_index);
}

} // namespace ET_MODULE_NAMESPACE
} // namespace extension
} // namespace executorch
