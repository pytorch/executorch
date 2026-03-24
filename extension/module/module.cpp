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
#include <executorch/extension/named_data_map/merged_data_map.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/platform/runtime.h>

namespace executorch {
namespace extension {
namespace ET_MODULE_NAMESPACE {

using ET_MERGED_DATA_MAP_NAMESPACE::MergedDataMap;
using ET_RUNTIME_NAMESPACE::MethodMeta;
using ET_RUNTIME_NAMESPACE::Program;

namespace {
runtime::Result<std::unique_ptr<runtime::DataLoader>> make_data_loader(
    const std::string& file_path,
    Module::LoadMode mode) {
  std::unique_ptr<runtime::DataLoader> data_loader;
  switch (mode) {
    case Module::LoadMode::File: {
      auto res = FileDataLoader::from(file_path.c_str());
      if (!res.ok()) {
        return res.error();
      }
      data_loader = std::make_unique<std::remove_reference_t<decltype(*res)>>(
          std::move(*res));
      break;
    }
    case Module::LoadMode::Mmap: {
      auto res_mmap = MmapDataLoader::from(
          file_path.c_str(), MmapDataLoader::MlockConfig::NoMlock);
      if (!res_mmap.ok()) {
        return res_mmap.error();
      }
      data_loader =
          std::make_unique<std::remove_reference_t<decltype(*res_mmap)>>(
              std::move(*res_mmap));
      break;
    }
    case Module::LoadMode::MmapUseMlock: {
      auto res_mlock = MmapDataLoader::from(file_path.c_str());
      if (!res_mlock.ok()) {
        return res_mlock.error();
      }
      data_loader =
          std::make_unique<std::remove_reference_t<decltype(*res_mlock)>>(
              std::move(*res_mlock));
      break;
    }
    case Module::LoadMode::MmapUseMlockIgnoreErrors: {
      auto res_mlock_ignore = MmapDataLoader::from(
          file_path.c_str(), MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
      if (!res_mlock_ignore.ok()) {
        return res_mlock_ignore.error();
      }
      data_loader = std::make_unique<
          std::remove_reference_t<decltype(*res_mlock_ignore)>>(
          std::move(*res_mlock_ignore));
      break;
    }
  }
  return data_loader;
}
} // namespace

Module::Module(
    const std::string& file_path,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    bool share_memory_arenas)
    : file_path_(file_path),
      load_mode_(load_mode),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)),
      share_memory_arenas_(share_memory_arenas) {
  runtime::runtime_init();
}

Module::Module(
    const std::string& file_path,
    const std::string& data_map_path,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    bool share_memory_arenas)
    : file_path_(file_path),
      load_mode_(load_mode),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)),
      share_memory_arenas_(share_memory_arenas) {
  if (!data_map_path.empty()) {
    data_files_.push_back(data_map_path);
  }
  runtime::runtime_init();
}

Module::Module(
    const std::string& file_path,
    std::vector<std::string> data_files,
    const LoadMode load_mode,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    bool share_memory_arenas)
    : file_path_(file_path),
      data_files_(std::move(data_files)),
      load_mode_(load_mode),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)),
      share_memory_arenas_(share_memory_arenas) {
  runtime::runtime_init();
}

Module::Module(
    std::unique_ptr<runtime::DataLoader> data_loader,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader,
    bool share_memory_arenas)
    : data_loader_(std::move(data_loader)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)),
      share_memory_arenas_(share_memory_arenas) {
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
    std::unique_ptr<runtime::DataLoader> data_map_loader,
    bool share_memory_arenas)
    : program_(std::move(program)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)),
      share_memory_arenas_(share_memory_arenas) {
  if (data_map_loader) {
    data_map_loaders_.push_back(std::move(data_map_loader));
  }
  runtime::runtime_init();
}

runtime::Error Module::load(const Program::Verification verification) {
  return load_internal(verification);
}

runtime::Error Module::load(
    const LoadBackendOptionsMap& backend_options,
    const Program::Verification verification) {
  backend_options_ = &backend_options;
  return load_internal(verification);
}

runtime::Error Module::load_internal(const Program::Verification verification) {
  if (!is_loaded()) {
    if (!data_loader_) {
      auto data_loader_result = make_data_loader(file_path_, load_mode_);
      if (!data_loader_result.ok()) {
        return data_loader_result.error();
      }
      data_loader_ = std::move(*data_loader_result);
    }
    if (data_files_.size() > 0) {
      for (const auto& data_file : data_files_) {
        auto data_map_loader_result = make_data_loader(data_file, load_mode_);
        if (!data_map_loader_result.ok()) {
          return data_map_loader_result.error();
        }
        data_map_loaders_.push_back(std::move(*data_map_loader_result));
      }
    }

    if (data_map_loaders_.size() > 0) {
      for (auto i = 0; i < data_map_loaders_.size(); ++i) {
        auto res_flat_tensor =
            FlatTensorDataMap::load(data_map_loaders_[i].get());
        if (!res_flat_tensor.ok()) {
          return res_flat_tensor.error();
        }
        named_data_maps_.push_back(
            std::make_unique<
                std::remove_reference_t<decltype(*res_flat_tensor)>>(
                std::move(*res_flat_tensor)));
      }

      // Extract raw pointers from unique_ptrs to pass to MergedDataMap::load()
      std::vector<const NamedDataMap*> raw_data_maps;
      raw_data_maps.reserve(named_data_maps_.size());
      for (const auto& data_map : named_data_maps_) {
        raw_data_maps.push_back(data_map.get());
      }
      auto res_merged = MergedDataMap::load(runtime::Span<const NamedDataMap*>(
          raw_data_maps.data(), raw_data_maps.size()));
      if (!res_merged.ok()) {
        return res_merged.error();
      }
      merged_data_map_ =
          std::make_unique<std::remove_reference_t<decltype(*res_merged)>>(
              std::move(*res_merged));
    }

    auto res_program = Program::load(data_loader_.get(), verification);
    if (!res_program.ok()) {
      return res_program.error();
    }
    auto program =
        std::make_unique<std::remove_reference_t<decltype(*res_program)>>(
            std::move(*res_program));
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

std::unique_ptr<Module::PlannedMemory> Module::make_planned_memory(
    const std::vector<size_t>& buffer_sizes) {
  auto planned = std::make_unique<PlannedMemory>();
  planned->planned_buffers.reserve(buffer_sizes.size());
  planned->planned_spans.reserve(buffer_sizes.size());
  for (size_t size : buffer_sizes) {
    planned->planned_buffers.emplace_back(size);
    planned->planned_spans.emplace_back(
        planned->planned_buffers.back().data(), size);
  }
  planned->planned_memory =
      std::make_unique<runtime::HierarchicalAllocator>(runtime::Span(
          planned->planned_spans.data(), planned->planned_spans.size()));
  return planned;
}

std::unique_ptr<Module::PlannedMemory>
Module::make_planned_memory_with_shared_arenas(
    const std::vector<size_t>& buffer_sizes,
    std::vector<std::vector<uint8_t>>& shared_arenas) {
  auto planned = std::make_unique<PlannedMemory>();
  planned->planned_buffers.reserve(buffer_sizes.size());
  planned->planned_spans.reserve(buffer_sizes.size());
  for (size_t i = 0; i < buffer_sizes.size(); i++) {
    if (i < shared_arenas.size()) {
      planned->planned_buffers.emplace_back();
      planned->planned_spans.emplace_back(
          shared_arenas[i].data(), shared_arenas[i].size());
    } else {
      planned->planned_buffers.emplace_back(buffer_sizes[i]);
      planned->planned_spans.emplace_back(
          planned->planned_buffers.back().data(), buffer_sizes[i]);
    }
  }
  planned->planned_memory =
      std::make_unique<runtime::HierarchicalAllocator>(runtime::Span(
          planned->planned_spans.data(), planned->planned_spans.size()));
  return planned;
}

std::unique_ptr<Module::PlannedMemory>
Module::make_planned_memory_with_devices(
    const ET_RUNTIME_NAMESPACE::MethodMeta& method_meta) {
  auto planned = std::make_unique<PlannedMemory>();
  const size_t num_buffers = method_meta.num_memory_planned_buffers();
  planned->planned_buffers.reserve(num_buffers);
  planned->planned_spans.reserve(num_buffers);

  for (size_t i = 0; i < num_buffers; ++i) {
    auto size = method_meta.memory_planned_buffer_size(i);
    ET_CHECK_MSG(size.ok(), "Failed to get buffer size for index %zu", i);
    auto device = method_meta.memory_planned_buffer_device(i);
    ET_CHECK_MSG(device.ok(), "Failed to get buffer device for index %zu", i);

    if (device->is_cpu()) {
      planned->planned_buffers.emplace_back(size.get());
      planned->planned_spans.emplace_back(
          planned->planned_buffers.back().data(), size.get());
    } else {
      // Allocate device memory via DeviceAllocator and store the RAII buffer.
      planned->planned_buffers.emplace_back(); // empty CPU placeholder
      auto dmb = runtime::DeviceMemoryBuffer::create(
          size.get(), device->type(), device->index());
      ET_CHECK_MSG(
          dmb.ok(),
          "Failed to allocate device memory for buffer %zu (device_type=%d)",
          i,
          static_cast<int>(device->type()));
      planned->planned_spans.emplace_back(dmb->as_span());
      planned->device_buffers.push_back(std::move(dmb.get()));
    }
  }

  planned->planned_memory =
      std::make_unique<runtime::HierarchicalAllocator>(runtime::Span(
          planned->planned_spans.data(), planned->planned_spans.size()));
  return planned;
}

runtime::Result<std::vector<size_t>> Module::get_mem_planned_buffer_sizes(
    const std::string& method_name) {
  auto meta_res = program_->method_meta(method_name.c_str());
  ET_CHECK_OK_OR_RETURN_ERROR(meta_res.error());
  auto meta = meta_res.get();
  std::vector<size_t> sizes;
  sizes.reserve(meta.num_memory_planned_buffers());
  for (size_t i = 0; i < meta.num_memory_planned_buffers(); i++) {
    auto size = meta.memory_planned_buffer_size(i);
    ET_CHECK_OK_OR_RETURN_ERROR(size.error());
    sizes.push_back(size.get());
  }
  return sizes;
}

runtime::Result<std::vector<size_t>>
Module::get_max_mem_planned_buffer_sizes() {
  std::vector<size_t> result;
  auto method_names_res = method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_res.error());
  for (const auto& name : method_names_res.get()) {
    auto sizes_res = get_mem_planned_buffer_sizes(name);
    ET_CHECK_OK_OR_RETURN_ERROR(sizes_res.error());
    auto& sizes = sizes_res.get();
    if (sizes.size() > result.size()) {
      result.resize(sizes.size(), 0);
    }
    for (size_t i = 0; i < sizes.size(); i++) {
      if (sizes[i] > result[i]) {
        result[i] = sizes[i];
      }
    }
  }
  return result;
}

runtime::Error Module::load_method(
    const std::string& method_name,
    runtime::HierarchicalAllocator* planned_memory,
    torch::executor::EventTracer* event_tracer,
    const LoadBackendOptionsMap* backend_options) {
  if (!is_method_loaded(method_name)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());

    // Use passed backend_options, or fall back to stored one from load()
    const LoadBackendOptionsMap* effective_backend_options =
        backend_options ? backend_options : backend_options_;

    MethodHolder method_holder;

    if (!planned_memory) {
      // Check if any buffers need device memory allocation.
      auto meta_res = program_->method_meta(method_name.c_str());
      ET_CHECK_OK_OR_RETURN_ERROR(meta_res.error());
      auto& meta = meta_res.get();

      bool has_device_buffers = false;
      for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
        auto dev = meta.memory_planned_buffer_device(i);
        if (dev.ok() && !dev->is_cpu()) {
          has_device_buffers = true;
          break;
        }
      }

      if (has_device_buffers) {
        // Device memory with shared arenas is not yet supported.
        ET_CHECK_OR_RETURN_ERROR(
            !share_memory_arenas_,
            NotSupported,
            "Device memory buffers are not yet compatible with "
            "share_memory_arenas. Please disable share_memory_arenas "
            "when using models with device-planned memory.");

        // Device-aware path: allocate CPU and device buffers, build metadata.
        method_holder.planned_memory =
            make_planned_memory_with_devices(meta);

        // Build per-buffer device type array for MemoryManager metadata.
        for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
          auto dev = meta.memory_planned_buffer_device(i);
          method_holder.buffer_devices.push_back(
              dev.ok() ? dev->type()
                       : runtime::etensor::DeviceType::CPU);
        }
        planned_memory = method_holder.planned_memory->planned_memory.get();

        method_holder.memory_manager = std::make_unique<runtime::MemoryManager>(
            memory_allocator_.get(),
            planned_memory,
            temp_allocator_.get(),
            runtime::Span<const runtime::etensor::DeviceType>(
                method_holder.buffer_devices.data(),
                method_holder.buffer_devices.size()));
      } else if (!share_memory_arenas_) {
        auto sizes_res = get_mem_planned_buffer_sizes(method_name);
        ET_CHECK_OK_OR_RETURN_ERROR(sizes_res.error());
        method_holder.planned_memory = make_planned_memory(sizes_res.get());
        planned_memory = method_holder.planned_memory->planned_memory.get();
      } else {
        auto sizes_res = get_mem_planned_buffer_sizes(method_name);
        ET_CHECK_OK_OR_RETURN_ERROR(sizes_res.error());
        auto& sizes = sizes_res.get();
        if (shared_arenas_.empty()) {
          auto max_res = get_max_mem_planned_buffer_sizes();
          ET_CHECK_OK_OR_RETURN_ERROR(max_res.error());
          auto& max_sizes = max_res.get();
          // Only share for mem_id=1,2.
          size_t shared = (max_sizes.size() > 2) ? 2 : max_sizes.size();
          for (size_t i = 0; i < shared; i++) {
            shared_arenas_.emplace_back(max_sizes[i]);
          }
        }
        method_holder.planned_memory =
            make_planned_memory_with_shared_arenas(sizes, shared_arenas_);
        planned_memory = method_holder.planned_memory->planned_memory.get();
      }
    }

    if (!method_holder.memory_manager) {
      method_holder.memory_manager = std::make_unique<runtime::MemoryManager>(
          memory_allocator_.get(), planned_memory, temp_allocator_.get());
    }
    auto res_method = program_->load_method(
        method_name.c_str(),
        method_holder.memory_manager.get(),
        event_tracer ? event_tracer : this->event_tracer(),
        merged_data_map_.get(),
        effective_backend_options);
    if (!res_method.ok()) {
      return res_method.error();
    }
    method_holder.method =
        std::make_unique<std::remove_reference_t<decltype(*res_method)>>(
            std::move(*res_method));
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
