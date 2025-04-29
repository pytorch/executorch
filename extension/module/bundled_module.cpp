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

namespace {
std::unique_ptr<BufferDataLoader> program_data_loader(
    const void* bundled_program_ptr) {
  auto bundled_program =
      bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr);
  // the program inside the bundled program
  auto program = bundled_program->program();
  return std::make_unique<BufferDataLoader>(program->data(), program->size());
}
} // namespace

BundledModule::BundledModule(
    const void* bundled_program_ptr,
    std::unique_ptr<runtime::MemoryAllocator> memory_allocator,
    std::unique_ptr<runtime::MemoryAllocator> temp_allocator,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    std::unique_ptr<runtime::DataLoader> data_map_loader)
    : Module(
          program_data_loader(bundled_program_ptr),
          std::move(memory_allocator),
          std::move(temp_allocator),
          std::move(event_tracer),
          std::move(data_map_loader)),
      bundled_program_ptr_(bundled_program_ptr) {}

runtime::Result<std::vector<runtime::EValue>> BundledModule::execute(
    const std::string& method_name,
    const size_t testset_idx) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;
  auto& inputs = methods_.at(method_name).inputs;

  ET_CHECK_OK_OR_RETURN_ERROR(
      executorch::BUNDLED_PROGRAM_NAMESPACE::load_bundled_input(
          *method, bundled_program_ptr_, testset_idx));
  ET_CHECK_OK_OR_RETURN_ERROR(method->get_inputs(inputs.data(), inputs.size()));
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

} // namespace extension
} // namespace executorch
