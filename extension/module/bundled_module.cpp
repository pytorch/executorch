/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/bundled_module.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/data_loader/shared_ptr_data_loader.h>

namespace executorch {
namespace extension {

namespace {
const flatbuffers::Vector<uint8_t>* embedded_program(
    const void* bundled_program_ptr) {
  return bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr)
      ->program();
}

std::unique_ptr<BufferDataLoader> program_data_loader(
    const void* bundled_program_ptr) {
  auto program = embedded_program(bundled_program_ptr);
  return std::make_unique<BufferDataLoader>(program->data(), program->size());
}
} // namespace

namespace ET_BUNDLED_MODULE_NAMESPACE {

BundledModule::BundledModule(
    const void* bundled_program_ptr,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader)
    : BundledModule(
          bundled_program_ptr,
          program_data_loader(bundled_program_ptr),
          std::move(memory_allocator),
          std::move(temp_allocator),
          std::move(event_tracer),
          std::move(data_map_loader)) {}

BundledModule::BundledModule(
    const void* bundled_program_ptr,
    std::unique_ptr<runtime::DataLoader> program_loader,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader)
    : Module(
          std::move(program_loader),
          std::move(memory_allocator),
          std::move(temp_allocator),
          std::move(event_tracer),
          std::move(data_map_loader)),
      bundled_program_ptr_(bundled_program_ptr) {}

runtime::Result<std::unique_ptr<BundledModule>> BundledModule::from_file(
    const std::string& file_path,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader) {
  auto data_loader_result = FileDataLoader::from(file_path.c_str());
  if (!data_loader_result.ok()) {
    return data_loader_result.error();
  }

  auto file_size_result = data_loader_result->size();
  if (!file_size_result.ok()) {
    return file_size_result.error();
  }

  size_t file_size = file_size_result.get();
  std::shared_ptr<uint8_t[]> file_data(new uint8_t[file_size]);
  auto buffer_result =
      data_loader_result->load_into(0, file_size, {}, file_data.get());
  if (buffer_result != runtime::Error::Ok) {
    return buffer_result;
  }

  // The loader shares the file data, so a Program it loads can outlive this
  // module.
  auto program = embedded_program(file_data.get());
  auto program_loader = std::make_unique<SharedPtrDataLoader>(
      std::shared_ptr<void>(file_data, const_cast<uint8_t*>(program->data())),
      program->size());
  std::unique_ptr<BundledModule> bm(new BundledModule(
      file_data.get(),
      std::move(program_loader),
      std::move(memory_allocator),
      std::move(temp_allocator),
      std::move(event_tracer),
      std::move(data_map_loader)));
  bm->file_data_ = std::move(file_data);

  return bm;
}

runtime::Result<std::vector<runtime::EValue>> BundledModule::execute(
    const std::string& method_name,
    const size_t testset_idx) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;

  ET_CHECK_OK_OR_RETURN_ERROR(
      executorch::BUNDLED_PROGRAM_NAMESPACE::load_bundled_input(
          *method, bundled_program_ptr_, testset_idx));
  ET_CHECK_OK_OR_RETURN_ERROR(method->execute());

  const auto outputs_size = method->outputs_size();
  std::vector<runtime::EValue> outputs(outputs_size);
  ET_CHECK_OK_OR_RETURN_ERROR(
      method->get_outputs(outputs.data(), outputs_size));

  return outputs;
}

runtime::Error BundledModule::verify_method_outputs(
    const std::string& method_name,
    const size_t testset_idx,
    double rtol,
    double atol) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  return executorch::BUNDLED_PROGRAM_NAMESPACE::verify_method_outputs(
      *method, bundled_program_ptr_, testset_idx, rtol, atol);
}
} // namespace ET_BUNDLED_MODULE_NAMESPACE
} // namespace extension
} // namespace executorch
