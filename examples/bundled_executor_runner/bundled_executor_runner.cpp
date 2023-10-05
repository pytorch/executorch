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
 * This tool can run ExecuTorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/bundled_program_schema_generated.h>
#include <executorch/util/bundled_program_verification.h>
#include <executorch/util/util.h>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4MB
static constexpr size_t kBundledAllocatorPoolSize = 16 * 1024U;
static uint8_t bundled_allocator_pool[kBundledAllocatorPoolSize];

DEFINE_string(
    bundled_program_path,
    "model_bundled.bp",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    prof_result_path,
    "prof_result.bin",
    "ExecuTorch profiler output path.");

DEFINE_int32(
    testset_idx,
    0,
    "Index of bundled verification set to be run "
    "by bundled model for verification");

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
  const char* bundled_program_path = FLAGS_bundled_program_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(bundled_program_path);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::from() failed: 0x%" PRIx32, loader.error());

  // Read in the entire file.
  Result<FreeableBuffer> file_data = loader->Load(0, loader->size().get());
  ET_CHECK_MSG(
      file_data.ok(),
      "Could not load contents of file '%s': 0x%x",
      bundled_program_path,
      (unsigned int)file_data.error());

  // Check whether the file is a bundled program.
  ET_CHECK_MSG(
      executorch_flatbuffer::BundledProgramBufferHasIdentifier(
          file_data->data()),
      "The file '%s' is not a bundled program.",
      bundled_program_path);

  // Find the offset to the embedded Program.
  const void* program_data;
  size_t program_data_len;
  Error status = torch::executor::util::GetProgramData(
      const_cast<void*>(file_data->data()),
      file_data->size(),
      &program_data,
      &program_data_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "GetProgramData() failed on file '%s': 0x%x",
      bundled_program_path,
      (unsigned int)status);

  auto buffer_data_loader =
      util::BufferDataLoader(program_data, program_data_len);

  // Parse the program file. This is immutable, and can also be reused
  // between multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&buffer_data_loader);
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", bundled_program_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", bundled_program_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%x",
      method_name,
      (unsigned int)method_meta.error());

  //
  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The method allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded method. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded method and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // method and seeing how much memory is actually used, though it's possible to
  // subclass MemoryAllocator so that it calls malloc() under the hood (see
  // MallocMemoryAllocator).
  //
  // In this example we use a statically allocated memory pool.
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  method_allocator.enable_profiling("method allocator");

  // The memory-planned buffers will back the mutable tensors used by the
  // method. The sizes of these buffers were determined ahead of time during the
  // memory-planning pasees.
  //
  // Each buffer typically corresponds to a different hardware memory bank. Most
  // mobile environments will only have a single buffer. Some embedded
  // environments may have more than one for, e.g., slow/large DRAM and
  // fast/small SRAM, or for memory associated with particular cores.
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  //
  // Load the method from the program, using the provided allocators. Running
  // the method can mutate the memory-planned buffers, so the method should only
  // be used by a single thread at at time, but it can be reused.
  //

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      method.error());
  ET_LOG(Info, "Method loaded.");

  // Prepare the inputs.
  // Use ones-initialized inputs or bundled inputs.
  MemoryAllocator bundled_input_allocator{
      MemoryAllocator(kBundledAllocatorPoolSize, bundled_allocator_pool)};
  exec_aten::ArrayRef<void*> inputs;
  // Use the inputs embedded in the bundled program.
  status = torch::executor::util::LoadBundledInput(
      *method,
      file_data->data(),
      &bundled_input_allocator,
      method_name,
      FLAGS_testset_idx);
  ET_CHECK_MSG(
      status == Error::Ok,
      "LoadBundledInput failed with status 0x%" PRIx32,
      status);

  ET_LOG(Info, "Inputs prepared.");

  // Run the model.
  status = method->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      status);
  ET_LOG(Info, "Model executed successfully.");

  // Print the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  for (EValue& output : outputs) {
    // TODO(T159700776): This assumes that all outputs are fp32 tensors. Add
    // support for other EValues and Tensor dtypes, and print tensors in a more
    // readable way.
    auto output_tensor = output.toTensor();
    auto data_output = output_tensor.const_data_ptr<float>();
    for (size_t j = 0; j < output_tensor.numel(); ++j) {
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

  // Verify the outputs.
  status = torch::executor::util::VerifyResultWithBundledExpectedOutput(
      *method,
      file_data->data(),
      &bundled_input_allocator,
      method_name,
      FLAGS_testset_idx,
      1e-5, // rtol
      1e-8 // atol
  );
  ET_CHECK_MSG(
      status == Error::Ok,
      "Bundle verification failed with status 0x%" PRIx32,
      status);
  ET_LOG(Info, "Model verified successfully.");

  return 0;
}
