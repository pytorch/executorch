/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

static uint8_t method_allocator_pool[1024];
static uint8_t activation_pool[512];

int main(int argc, char** argv) {
  runtime_init();

  ET_CHECK_MSG(argc == 2, "Expected model file argument.");

  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);
  method_allocator.enable_profiling("method allocator");

  Span<uint8_t> memory_planned_buffers[1]{
      {activation_pool, sizeof(activation_pool)}};
  HierarchicalAllocator planned_memory({memory_planned_buffers, 1});

  MemoryManager memory_manager(&method_allocator, &planned_memory);

  Result<FileDataLoader> loader = FileDataLoader::from(argv[1]);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      static_cast<uint32_t>(loader.error()));

  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");
  const auto program = Program::load(&loader.get());
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK_MSG(
      program.ok(),
      "Program::load() failed: 0x%" PRIx32,
      static_cast<uint32_t>(program.error()));
  ET_LOG(Info, "Program file %s loaded.", argv[1]);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Loading method %s", method_name);

  prof_tok = EXECUTORCH_BEGIN_PROF("load model");
  Result<Method> method = program->load_method(method_name, &memory_manager);
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(method.ok());

  ET_LOG(Info, "Method loaded.");

  // Prepare for inputs
  // It assumes the input is one tensor.
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Tensor::SizesType sizes[] = {6};
  Tensor::DimOrderType dim_order[] = {0};
  TensorImpl impl(ScalarType::Float, 1, sizes, data, dim_order);
  Tensor t(&impl);
  Error set_input_error = method->set_input(t, 0);
  ET_CHECK(set_input_error == Error::Ok);

  ET_LOG(Info, "Inputs prepared.");

  prof_tok = EXECUTORCH_BEGIN_PROF("run model");
  Error status = method->execute();
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(status == Error::Ok);
  ET_LOG(Info, "Model executed successfully.");

  // print output
  auto output_list =
      method_allocator.allocateList<EValue>(method->outputs_size());

  status = method->get_outputs(output_list, method->outputs_size());
  ET_CHECK(status == Error::Ok);

  // It assumes the outputs are all tensors.
  for (size_t i = 0; i < method->outputs_size(); i++) {
    auto output_tensor = output_list[i].toTensor();
    auto data_output = output_tensor.const_data_ptr<float>();
    for (size_t j = 0; j < output_list[i].toTensor().numel(); ++j) {
      ET_LOG(Info, "%f", data_output[j]);
    }
  }
  prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen("prof_result.bin", "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  return 0;
}
