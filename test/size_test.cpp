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
#include <executorch/util/util.h>
#include <stdio.h>

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

static constexpr size_t kRuntimeMemorySize = 1024;
static constexpr size_t kMemoryAmount = 512;

static uint8_t runtime_pool[kRuntimeMemorySize];
static uint8_t activation_pool[kMemoryAmount];

int main(int argc, char** argv) {
  runtime_init();

  ET_CHECK_MSG(argc == 2, "Expected model file argument.");

  MemoryAllocator const_allocator{MemoryAllocator(0, nullptr)};
  const_allocator.enable_profiling("const allocator");

  MemoryAllocator runtime_allocator{
      MemoryAllocator(kRuntimeMemorySize, runtime_pool)};
  runtime_allocator.enable_profiling("runtime allocator");

  MemoryAllocator temp_allocator{MemoryAllocator(0, nullptr)};
  temp_allocator.enable_profiling("temp allocator");

  Span<uint8_t> non_const_buffers[1]{{activation_pool, kMemoryAmount}};
  HierarchicalAllocator non_const_allocator({non_const_buffers, 1});

  MemoryManager memory_manager{MemoryManager(
      &const_allocator,
      &non_const_allocator,
      &runtime_allocator,
      &temp_allocator)};

  Result<FileDataLoader> loader = FileDataLoader::from(argv[1]);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::from() failed: 0x%" PRIx32, loader.error());

  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");
  const auto program = Program::load(&loader.get());
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK_MSG(
      program.ok(), "Program::load() failed: 0x%" PRIx32, program.error());
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
  auto inputs = torch::executor::util::PrepareInputTensors(*method);

  ET_LOG(Info, "Inputs prepared.");

  prof_tok = EXECUTORCH_BEGIN_PROF("run model");
  Error status = method->execute();
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(status == Error::Ok);
  ET_LOG(Info, "Model executed successfully.");

  // print output
  auto output_list =
      runtime_allocator.allocateList<EValue>(method->outputs_size());

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
  torch::executor::util::FreeInputs(inputs);
  prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen("prof_result.bin", "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  return 0;
}
