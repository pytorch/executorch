/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run Executorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>

static constexpr size_t kRuntimeMemorySize = 4 * 1024U * 1024U; // 4 MB
static uint8_t runtime_pool[kRuntimeMemorySize];

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    prof_result_path,
    "prof_result.bin",
    "Executorch profiler output path.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

int main(int argc, char** argv) {
  runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::From(model_path);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::From() failed: 0x%" PRIx32, loader.error());

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::Load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const size_t plan_index = 0;
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(plan_index);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", method_name);

  //
  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The runtime allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded program. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded program and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // program and seeing how much memory is actually used, though it's possible
  // to subclass MemoryAllocator so that it calls malloc() under the hood.

  // In this example we using statically allocated gloabl runtime_pool of
  // size kRuntimeMemorySize
  MemoryAllocator runtime_allocator{
      MemoryAllocator(kRuntimeMemorySize, runtime_pool)};
  runtime_allocator.enable_profiling("runtime allocator");

  // The non-const allocator is used to provide the memory-planned buffers that
  // back mutable tensors. Since it was planned ahead of time, the Program knows
  // how big each of the allocators needs to be.
  //
  // These buffers correspond to different hardware memory banks. Most mobile
  // environments will only have a single buffer. Some embedded environments may
  // have more than one for, e.g., slow/large DRAM and fast/small SRAM.
  std::vector<std::unique_ptr<uint8_t[]>> non_const_buffers;
  std::vector<MemoryAllocator> non_const_allocators;
  size_t num_non_const_buffers = 0;
  {
    auto result = program->num_non_const_buffers(method_name);
    ET_CHECK_MSG(
        result.ok(),
        "Failed to get number of non-const buffers for method %s: 0x%x",
        method_name,
        (unsigned int)result.error());
    num_non_const_buffers = *result;
  }
  // Note that this loop starts at ID 1, because ID 0 is reserved. But, the
  // HierarchicalAllocator indices are zero-based, so it's later adjusted by -1.
  for (size_t id = 1; id < num_non_const_buffers; ++id) {
    auto buffer_size = program->get_non_const_buffer_size(id, method_name);
    ET_CHECK_MSG(
        buffer_size.ok(),
        "Failed to get size of non-const buffer %zu for method %s: 0x%x",
        id,
        method_name,
        (unsigned int)buffer_size.error());
    ET_LOG(
        Info, "Setting up non-const buffer %zu, size %zu.", id, *buffer_size);
    non_const_buffers.push_back(std::make_unique<uint8_t[]>(*buffer_size));
    // Since the list of allocators began empty, buffer ID N will live at index
    // N-1.
    non_const_allocators.push_back(
        MemoryAllocator(*buffer_size, non_const_buffers.back().get()));
    non_const_allocators.back().enable_profiling("non_const_allocators");
  }
  HierarchicalAllocator non_const_allocator(
      non_const_allocators.size(), non_const_allocators.data());

  // The constant allocator is not currently used. Please initialize with a
  // zero-sized allocator.
  MemoryAllocator const_allocator{MemoryAllocator(0, nullptr)};
  const_allocator.enable_profiling("const allocator");

  // The kernel temporary allocator is not currently used. Please initialize
  // with a zero-sized allocator.
  MemoryAllocator temp_allocator{MemoryAllocator(0, nullptr)};
  temp_allocator.enable_profiling("temp allocator");

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(
      &const_allocator,
      &non_const_allocator,
      &runtime_allocator,
      &temp_allocator);

  //
  // Load method from the program, using the provided
  // allocators. Running the method can mutate allocated non_const buffers,
  // so should only be used by a single thread at at time, but it can be reused.
  //

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      method.error());
  ET_LOG(Info, "Method loaded.");

  // Prepare the inputs.
  // Use ones-initialized inputs.
  auto inputs = util::PrepareInputTensors(*method);
  ET_LOG(Info, "Inputs prepared.");

  // Run the model.
  Error status = method->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      status);
  ET_LOG(Info, "Model executed successfully.");

  auto output_list =
      runtime_allocator.allocateList<EValue>(method->outputs_size());
  status = method->get_outputs(output_list, method->outputs_size());
  ET_CHECK(status == Error::Ok);
  // The following code assumes all output EValues are floating point
  // tensors. We need to handle other types of EValues and tensor
  // dtypes. Furthermore, we need a util to print tensors in a more
  // interpretable (e.g. size, dtype) and readable way.
  // TODO for the above at T159700776
  for (size_t i = 0; i < method->outputs_size(); i++) {
    auto output_tensor = output_list[i].toTensor();
    auto data_output = output_tensor.const_data_ptr<float>();
    for (size_t j = 0; j < output_list[i].toTensor().numel(); ++j) {
      ET_LOG(Info, "%f", data_output[j]);
    }
  }

  // Dump the profiling data to the specified file.
  torch::executor::prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen(FLAGS_prof_result_path.c_str(), "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  util::FreeInputs(inputs);
  return 0;
}
