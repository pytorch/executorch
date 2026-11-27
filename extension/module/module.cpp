/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <algorithm>
#include <functional>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/module/ptn_module.h>
#include <executorch/extension/named_data_map/merged_data_map.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/platform/runtime.h>

namespace executorch {
namespace extension {
namespace ET_MODULE_NAMESPACE {

using ET_MERGED_DATA_MAP_NAMESPACE::MergedDataMap;
using ET_RUNTIME_NAMESPACE::Kernel;
using ET_RUNTIME_NAMESPACE::MethodMeta;
using ET_RUNTIME_NAMESPACE::Program;

namespace {
runtime::Result<bool> is_ptn_source(runtime::DataLoader& loader) {
  auto size = loader.size();
  if (!size.ok()) {
    return size.error();
  }
  if (*size < 2) {
    return false;
  }
  // DataLoader requires a segment kind; this prefix belongs to the top-level
  // program artifact.
  auto head = loader.load(
      /*offset=*/0,
      /*size=*/2,
      runtime::DataLoader::SegmentInfo(
          runtime::DataLoader::SegmentInfo::Type::Program));
  if (!head.ok()) {
    return head.error();
  }
  auto data = head->data_safe();
  if (!data.ok()) {
    return data.error();
  }
  if (head->size() < 2 || *data == nullptr) {
    return runtime::Error::InvalidProgram;
  }
  const auto* bytes = static_cast<const uint8_t*>(*data);
  // PTN packages are ZIP archives, whose signatures begin with "PK". The PTN
  // provider performs full package validation after this routing check.
  return bytes[0] == 'P' && bytes[1] == 'K';
}

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
    case Module::LoadMode::MmapUseMadvise: {
      auto res_madvise = MmapDataLoader::from(
          file_path.c_str(), MmapDataLoader::MlockConfig::UseMadvise);
      if (!res_madvise.ok()) {
        return res_madvise.error();
      }
      data_loader =
          std::make_unique<std::remove_reference_t<decltype(*res_madvise)>>(
              std::move(*res_madvise));
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
  if (ptn_ && backend_options.size() != 0) {
    return runtime::Error::NotSupported;
  }
  // load_internal does not read backend options, so run it first; on
  // failure we skip the deep-copy work entirely and leave the prior
  // installed options (if any) in place.
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_internal(verification, backend_options.size() != 0));

  // Deep-copy the input into local storage so the Module owns the
  // BackendOption arrays for the lifetime of any methods loaded with
  // these options. Build BOTH the storage and the map in locals so any
  // mid-loop failure (or exception from emplace) leaves the prior
  // installed state untouched -- the two members are only committed
  // together at the end on full success.
  //
  // local_storage is reserve()'d up front, so emplace_back() never
  // reallocates the outer buffer and the inner vectors keep stable
  // addresses while we build local_map. The final move of
  // local_storage into backend_options_storage_ uses std::vector's
  // O(1) buffer transfer, so the heap buffers that local_map's spans
  // point into remain valid after the move; the static_assert documents
  // the inner-vector property we rely on for that span stability.
  static_assert(
      std::is_nothrow_move_constructible_v<std::vector<runtime::BackendOption>>,
      "Moving local_storage must not move-construct the inner vectors; "
      "local_map's spans reference their heap buffers.");

  std::vector<std::vector<runtime::BackendOption>> local_storage;
  local_storage.reserve(backend_options.size());
  LoadBackendOptionsMap local_map;
  for (size_t i = 0; i < backend_options.size(); ++i) {
    const auto entry = backend_options.entry_at(i);
    local_storage.emplace_back(entry.options.begin(), entry.options.end());
    auto& owned = local_storage.back();
    // The input map was already valid, so set_options should not fail
    // here; assert it loudly rather than leaving partial state behind.
    ET_CHECK_OK_OR_RETURN_ERROR(local_map.set_options(
        entry.backend_id,
        runtime::Span<runtime::BackendOption>(owned.data(), owned.size())));
  }

  // Single commit point: both members updated together.
  backend_options_storage_ = std::move(local_storage);
  backend_options_map_ = std::move(local_map);

  return runtime::Error::Ok;
}

runtime::Error Module::load_internal(
    const Program::Verification verification,
    bool has_backend_options) {
  if (!is_loaded()) {
    if (!data_loader_) {
      auto data_loader_result = make_data_loader(file_path_, load_mode_);
      if (!data_loader_result.ok()) {
        return data_loader_result.error();
      }
      data_loader_ = std::move(*data_loader_result);
    }
    auto ptn_detection_result = is_ptn_source(*data_loader_);
    if (!ptn_detection_result.ok()) {
      return ptn_detection_result.error();
    }
    if (*ptn_detection_result) {
      if (load_mode_ != LoadMode::File && load_mode_ != LoadMode::Mmap) {
        return runtime::Error::NotSupported;
      }
      if (has_backend_options) {
        return runtime::Error::NotSupported;
      }
      if (!data_files_.empty() || !data_map_loaders_.empty()) {
        return runtime::Error::InvalidArgument;
      }
      const ET_PTN_MODULE_NAMESPACE::internal::PtnSource source =
          file_path_.empty()
          ? ET_PTN_MODULE_NAMESPACE::internal::PtnSource(
                std::ref(*data_loader_))
          : ET_PTN_MODULE_NAMESPACE::internal::PtnSource(
                ET_PTN_MODULE_NAMESPACE::internal::PtnFileSource{
                    file_path_,
                    load_mode_ == LoadMode::Mmap
                        ? ET_PTN_MODULE_NAMESPACE::internal::PtnFileSource::
                              Mode::Mmap
                        : ET_PTN_MODULE_NAMESPACE::internal::PtnFileSource::
                              Mode::Read});
      auto ptn_load_result =
          ET_PTN_MODULE_NAMESPACE::load_ptn(source, verification);
      if (!ptn_load_result.ok()) {
        return ptn_load_result.error();
      }
      ptn_ = std::move(*ptn_load_result);
      return runtime::Error::Ok;
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
  if (ptn_) {
    return ptn_->num_methods();
  }
  return program_->num_methods();
}

runtime::Result<std::unordered_set<std::string>> Module::method_names() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  if (ptn_) {
    return ptn_->method_names();
  }
  const auto method_count = program_->num_methods();
  std::unordered_set<std::string> result;
  result.reserve(method_count);

  for (auto index = 0; index < method_count; ++index) {
    result.emplace(program_->get_method_name(index).get());
  }
  return result;
}

runtime::Result<runtime::Span<uint8_t>> Module::get_shared_buffer(
    size_t index,
    runtime::etensor::Device device) {
  auto found = shared_buffers_.find(index);
  if (found == shared_buffers_.end()) {
    // Size the allocation for every method that shares it, so a method loaded
    // later never needs it to grow under the methods already using it.
    int64_t size = 0;
    std::vector<const char*> sharing_methods;
    // Program order, as the Python runtime uses.
    for (size_t i = 0; i < program_->num_methods(); ++i) {
      auto name = program_->get_method_name(i);
      ET_CHECK_OK_OR_RETURN_ERROR(name.error());
      auto meta = program_->method_meta(name.get());
      ET_CHECK_OK_OR_RETURN_ERROR(meta.error());
      if (index >= meta->num_memory_planned_buffers()) {
        continue;
      }
      auto shared = meta->memory_planned_buffer_is_shared(index);
      ET_CHECK_OK_OR_RETURN_ERROR(shared.error());
      if (!shared.get()) {
        continue;
      }
      auto other_device = meta->memory_planned_buffer_device(index);
      ET_CHECK_OK_OR_RETURN_ERROR(other_device.error());
      ET_CHECK_OR_RETURN_ERROR(
          other_device.get() == device,
          InvalidProgram,
          "Methods place shared buffer %zu on different devices",
          index);
      auto other_size = meta->memory_planned_buffer_size(index);
      ET_CHECK_OK_OR_RETURN_ERROR(other_size.error());
      size = std::max(size, other_size.get());
      sharing_methods.push_back(name.get());
    }
    SharedBuffer buffer;
    if (device.is_cpu()) {
      buffer.host_buffer.resize(size);
      buffer.span = {buffer.host_buffer.data(), buffer.host_buffer.size()};
    } else {
      auto device_buffer = runtime::DeviceMemoryBuffer::create(
          size, device.type(), device.index());
      ET_CHECK_OK_OR_RETURN_ERROR(
          device_buffer.error(),
          "Failed to allocate device memory for shared buffer %zu (device_type=%d, device_index=%d)",
          index,
          static_cast<int>(device.type()),
          static_cast<int>(device.index()));
      buffer.device_buffer = std::move(device_buffer.get());
      buffer.span = buffer.device_buffer.as_span();
    }
    // Initialize it for every method that shares it now, once, so that
    // loading a method later never rewinds state another method has changed.
    std::vector<runtime::Span<uint8_t>> spans(index + 1);
    spans[index] = buffer.span;
    std::vector<runtime::etensor::Device> devices(
        index + 1,
        runtime::etensor::Device{runtime::etensor::DeviceType::CPU, 0});
    devices[index] = device;
    runtime::HierarchicalAllocator allocator(
        {spans.data(), spans.size()}, {devices.data(), devices.size()});
    for (const char* name : sharing_methods) {
      ET_CHECK_OK_OR_RETURN_ERROR(program_->initialize_planned_buffer(
          name, index, &allocator, merged_data_map_.get()));
    }
    found = shared_buffers_.emplace(index, std::move(buffer)).first;
  }
  return found->second.span;
}

runtime::Result<std::unique_ptr<Module::PlannedMemory>>
Module::make_planned_memory(
    const ET_RUNTIME_NAMESPACE::MethodMeta& method_meta) {
  auto planned = std::make_unique<PlannedMemory>();
  const size_t num_buffers = method_meta.num_memory_planned_buffers();
  planned->planned_buffers.reserve(num_buffers);
  planned->planned_spans.reserve(num_buffers);
  planned->device_buffers.reserve(num_buffers);
  planned->planned_devices.reserve(num_buffers);
  planned->initialized_buffers = std::make_unique<bool[]>(num_buffers);

  if (share_memory_arenas_) {
    // A program that marks no buffer relies on the host buffers at mem_id 1
    // and 2 being shared, so state it plans in device memory would stay
    // private.
    bool marks_shared_buffers = false;
    auto method_names_res = method_names();
    ET_CHECK_OK_OR_RETURN_ERROR(method_names_res.error());
    for (const auto& name : method_names_res.get()) {
      auto meta = program_->method_meta(name.c_str());
      ET_CHECK_OK_OR_RETURN_ERROR(meta.error());
      for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
        auto shared = meta->memory_planned_buffer_is_shared(i);
        ET_CHECK_OK_OR_RETURN_ERROR(shared.error());
        marks_shared_buffers |= shared.get();
      }
    }
    for (size_t i = 0; !marks_shared_buffers && i < num_buffers; ++i) {
      auto device = method_meta.memory_planned_buffer_device(i);
      ET_CHECK_OK_OR_RETURN_ERROR(device.error());
      ET_CHECK_OR_RETURN_ERROR(
          device->is_cpu(),
          NotSupported,
          "share_memory_arenas does not work with device memory in a program "
          "that does not mark its shared buffers. Export it with "
          "share_mutable_buffers=True, or disable share_memory_arenas.");
    }
  }

  bool has_device_buffers = false;
  bool has_shared_buffers = false;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto size = method_meta.memory_planned_buffer_size(i);
    ET_CHECK_OK_OR_RETURN_ERROR(
        size.error(), "Failed to get buffer size for index %zu", i);
    // A failed device query must not read as "this buffer is on the host",
    // which would silently hand a backend host memory.
    auto device = method_meta.memory_planned_buffer_device(i);
    ET_CHECK_OK_OR_RETURN_ERROR(
        device.error(), "Failed to get buffer device for index %zu", i);
    auto shared = method_meta.memory_planned_buffer_is_shared(i);
    ET_CHECK_OK_OR_RETURN_ERROR(
        shared.error(), "Failed to get buffer sharing for index %zu", i);
    planned->planned_devices.push_back(device.get());
    has_device_buffers |= !device->is_cpu();

    if (shared.get()) {
      auto span = get_shared_buffer(i, device.get());
      ET_CHECK_OK_OR_RETURN_ERROR(span.error());
      planned->planned_buffers.emplace_back();
      planned->planned_spans.push_back(span.get());
      planned->initialized_buffers[i] = true;
      has_shared_buffers = true;
    } else if (device->is_cpu() && share_memory_arenas_ && i < 2) {
      if (shared_arenas_.empty()) {
        auto max_res = get_max_mem_planned_buffer_sizes();
        ET_CHECK_OK_OR_RETURN_ERROR(max_res.error());
        auto& max_sizes = max_res.get();
        // Only share for mem_id=1,2.
        size_t shared_count = (max_sizes.size() > 2) ? 2 : max_sizes.size();
        for (size_t j = 0; j < shared_count; j++) {
          shared_arenas_.emplace_back(max_sizes[j]);
        }
      }
      planned->planned_buffers.emplace_back();
      planned->planned_spans.emplace_back(
          shared_arenas_[i].data(), shared_arenas_[i].size());
    } else if (device->is_cpu()) {
      planned->planned_buffers.emplace_back(size.get());
      planned->planned_spans.emplace_back(
          planned->planned_buffers.back().data(), size.get());
    } else {
      // Allocate device memory via DeviceAllocator and store the RAII buffer.
      planned->planned_buffers.emplace_back(); // empty CPU placeholder
      // Whether a device allocator exists, and whether the device has room,
      // are properties of the machine rather than of the program, so report
      // them to the caller instead of aborting the process.
      auto dmb = runtime::DeviceMemoryBuffer::create(
          size.get(), device->type(), device->index());
      ET_CHECK_OK_OR_RETURN_ERROR(
          dmb.error(),
          "Failed to allocate device memory for buffer %zu (device_type=%d, device_index=%d)",
          i,
          static_cast<int>(device->type()),
          static_cast<int>(device->index()));
      planned->planned_spans.emplace_back(dmb->as_span());
      planned->device_buffers.push_back(std::move(dmb.get()));
    }
  }

  runtime::Span<runtime::Span<uint8_t>> spans(
      planned->planned_spans.data(), planned->planned_spans.size());
  // HierarchicalAllocator owns the per-buffer Device metadata so the
  // MemoryManager can later expose it via planned_buffer_devices().
  planned->planned_memory = has_device_buffers
      ? std::make_unique<runtime::HierarchicalAllocator>(
            spans,
            runtime::Span<const runtime::etensor::Device>(
                planned->planned_devices.data(),
                planned->planned_devices.size()))
      : std::make_unique<runtime::HierarchicalAllocator>(spans);
  if (has_shared_buffers) {
    planned->planned_memory->set_initialized_buffers(
        {planned->initialized_buffers.get(), num_buffers});
  }
  return planned;
}

runtime::Result<std::vector<size_t>> Module::get_mem_planned_buffer_sizes(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  if (ptn_) {
    return runtime::Error::NotSupported;
  }
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
    auto meta = program_->method_meta(name.c_str());
    ET_CHECK_OK_OR_RETURN_ERROR(meta.error());
    if (sizes.size() > result.size()) {
      result.resize(sizes.size(), 0);
    }
    for (size_t i = 0; i < sizes.size(); i++) {
      // Device and shared buffers never come from the host arenas this sizes.
      auto device = meta->memory_planned_buffer_device(i);
      ET_CHECK_OK_OR_RETURN_ERROR(device.error());
      auto shared = meta->memory_planned_buffer_is_shared(i);
      ET_CHECK_OK_OR_RETURN_ERROR(shared.error());
      if (device->is_cpu() && !shared.get() && sizes[i] > result[i]) {
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
    const LoadBackendOptionsMap* backend_options,
    std::vector<Kernel> kernel_registry) {
  if (!is_method_loaded(method_name)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    if (ptn_) {
      if (planned_memory != nullptr || event_tracer != nullptr ||
          (backend_options != nullptr && backend_options->size() != 0) ||
          !kernel_registry.empty()) {
        return runtime::Error::NotSupported;
      }
      return ptn_->load_method(method_name);
    }

    // Use passed backend_options, or fall back to stored ones from load().
    // An empty stored map behaves identically to nullptr downstream, so we
    // only forward the stored map when it actually has entries.
    const LoadBackendOptionsMap* effective_backend_options = backend_options
        ? backend_options
        : (backend_options_map_.size() > 0 ? &backend_options_map_ : nullptr);

    MethodHolder method_holder;

    if (!planned_memory) {
      auto meta_res = program_->method_meta(method_name.c_str());
      ET_CHECK_OK_OR_RETURN_ERROR(meta_res.error());
      auto planned_res = make_planned_memory(meta_res.get());
      ET_CHECK_OK_OR_RETURN_ERROR(planned_res.error());
      method_holder.planned_memory = std::move(planned_res.get());
      planned_memory = method_holder.planned_memory->planned_memory.get();
    }

    method_holder.memory_manager = std::make_unique<runtime::MemoryManager>(
        memory_allocator_.get(), planned_memory, temp_allocator_.get());
    method_holder.kernel_registry = std::move(kernel_registry);
    auto res_method = program_->load_method(
        method_name.c_str(),
        method_holder.memory_manager.get(),
        event_tracer ? event_tracer : this->event_tracer(),
        merged_data_map_.get(),
        effective_backend_options,
        runtime::Span<const Kernel>(
            method_holder.kernel_registry.data(),
            method_holder.kernel_registry.size()));
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
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  if (ptn_) {
    return runtime::Error::NotSupported;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  return methods_[method_name].method.get();
}

runtime::Result<MethodMeta> Module::method_meta(
    const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  if (ptn_) {
    return ptn_->method_meta(method_name);
  }
  return program_->method_meta(method_name.c_str());
}

runtime::Result<std::vector<runtime::EValue>> Module::execute(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  if (ptn_) {
    return ptn_->execute(method_name, input_values);
  }
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
  if (ptn_) {
    return ptn_->set_input(method_name, input_value, input_index);
  }
  auto& method = methods_.at(method_name).method;
  return method->set_input(input_value, input_index);
}

runtime::Error Module::set_inputs(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input_values) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  if (ptn_) {
    return ptn_->set_inputs(method_name, input_values);
  }
  auto& method = methods_.at(method_name).method;
  return method->set_inputs(executorch::aten::ArrayRef<runtime::EValue>(
      input_values.data(), input_values.size()));
}

runtime::Error Module::set_output(
    const std::string& method_name,
    runtime::EValue output_value,
    size_t output_index) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  if (ptn_) {
    return ptn_->set_output(method_name, std::move(output_value), output_index);
  }
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
  if (ptn_) {
    return ptn_->set_outputs(method_name, output_values);
  }
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
  if (ptn_) {
    return ptn_->get_outputs(method_name);
  }
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
  if (ptn_) {
    return ptn_->get_output(method_name, output_index);
  }
  auto& method = methods_.at(method_name).method;
  ET_CHECK_OR_RETURN_ERROR(
      output_index < method->outputs_size(),
      InvalidArgument,
      "output index: %zu is out of range",
      output_index);
  return method->get_output(output_index);
}

runtime::Result<Module::Format> Module::format() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  return ptn_ ? Format::Ptn : Format::Pte;
}

bool Module::unload_method(const std::string& method_name) {
  return ptn_ ? ptn_->unload_method(method_name) : methods_.erase(method_name);
}

bool Module::is_method_loaded(const std::string& method_name) const {
  return ptn_ ? ptn_->is_method_loaded(method_name)
              : methods_.count(method_name) != 0;
}

} // namespace ET_MODULE_NAMESPACE
} // namespace extension
} // namespace executorch
