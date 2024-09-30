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

#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

static std::array<uint8_t, 4 * 1024U * 1024U> method_allocator_pool; // 4MB

DEFINE_string(
    bundled_program_path,
    "model_bundled.bpte",
    "Model serialized in flatbuffer format.");

DEFINE_int32(
    testset_idx,
    0,
    "Index of bundled verification set to be run "
    "by bundled model for verification");

DEFINE_string(
    etdump_path,
    "etdump.etdp",
    "If etdump generation is enabled an etdump will be written out to this path");

DEFINE_bool(
    output_verification,
    false,
    "Comapre the model output to the reference outputs present in the BundledProgram.");

DEFINE_bool(
    print_output,
    false,
    "Print the output of the ET model to stdout, if needs.");

DEFINE_bool(dump_outputs, false, "Dump outputs to etdump file");

DEFINE_bool(
    dump_intermediate_outputs,
    false,
    "Dump intermediate outputs to etdump file.");

DEFINE_string(
    debug_output_path,
    "debug_output.bin",
    "Path to dump debug outputs to.");

DEFINE_int32(
    debug_buffer_size,
    262144, // 256 KB
    "Size of the debug buffer in bytes to allocate for intermediate outputs and program outputs logging.");

using executorch::etdump::ETDumpGen;
using executorch::etdump::ETDumpResult;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

std::vector<uint8_t> load_file_or_die(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const size_t nbytes = file.tellg();
  file.seekg(0, std::ios::beg);
  auto file_data = std::vector<uint8_t>(nbytes);
  ET_CHECK_MSG(
      file.read(reinterpret_cast<char*>(file_data.data()), nbytes),
      "Could not load contents of file '%s'",
      path);
  return file_data;
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Read in the entire file.
  const char* bundled_program_path = FLAGS_bundled_program_path.c_str();
  std::vector<uint8_t> file_data = load_file_or_die(bundled_program_path);

  // Find the offset to the embedded Program.
  const void* program_data;
  size_t program_data_len;
  Error status = executorch::bundled_program::get_program_data(
      reinterpret_cast<void*>(file_data.data()),
      file_data.size(),
      &program_data,
      &program_data_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "get_program_data() failed on file '%s': 0x%x",
      bundled_program_path,
      (unsigned int)status);

  auto buffer_data_loader = BufferDataLoader(program_data, program_data_len);

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
  MemoryAllocator method_allocator{MemoryAllocator(
      sizeof(method_allocator_pool), method_allocator_pool.data())};

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
  ETDumpGen etdump_gen;
  Result<Method> method =
      program->load_method(method_name, &memory_manager, &etdump_gen);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      method.error());
  ET_LOG(Info, "Method loaded.");

  void* debug_buffer = malloc(FLAGS_debug_buffer_size);
  if (FLAGS_dump_intermediate_outputs) {
    Span<uint8_t> buffer((uint8_t*)debug_buffer, FLAGS_debug_buffer_size);
    etdump_gen.set_debug_buffer(buffer);
    etdump_gen.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kIntermediateOutputs);
  } else if (FLAGS_dump_outputs) {
    Span<uint8_t> buffer((uint8_t*)debug_buffer, FLAGS_debug_buffer_size);
    etdump_gen.set_debug_buffer(buffer);
    etdump_gen.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kProgramOutputs);
  }
  // Use the inputs embedded in the bundled program.
  status = executorch::bundled_program::load_bundled_input(
      *method, file_data.data(), FLAGS_testset_idx);
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
  if (FLAGS_print_output) {
    std::vector<EValue> outputs(method->outputs_size());
    status = method->get_outputs(outputs.data(), outputs.size());
    ET_CHECK(status == Error::Ok);
    for (EValue& output : outputs) {
      // TODO(T159700776): This assumes that all outputs are fp32 tensors. Add
      // support for other EValues and Tensor dtypes, and print tensors in a
      // more readable way.
      auto output_tensor = output.toTensor();
      auto data_output = output_tensor.const_data_ptr<float>();
      for (size_t j = 0; j < output_tensor.numel(); ++j) {
        ET_LOG(Info, "%f", data_output[j]);
      }
    }
  }

  // Dump the etdump data containing profiling/debugging data to the specified
  // file.
  ETDumpResult result = etdump_gen.get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    FILE* f = fopen(FLAGS_etdump_path.c_str(), "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  }

  if (FLAGS_output_verification) {
    // Verify the outputs.
    status = executorch::bundled_program::verify_method_outputs(
        *method,
        file_data.data(),
        FLAGS_testset_idx,
        1e-3, // rtol
        1e-5 // atol
    );
    ET_CHECK_MSG(
        status == Error::Ok,
        "Bundle verification failed with status 0x%" PRIx32,
        status);
    ET_LOG(Info, "Model verified successfully.");
  }

  if (FLAGS_dump_outputs || FLAGS_dump_intermediate_outputs) {
    FILE* f = fopen(FLAGS_debug_output_path.c_str(), "w+");
    fwrite((uint8_t*)debug_buffer, 1, FLAGS_debug_buffer_size, f);
    fclose(f);
  }
  free(debug_buffer);

  return 0;
}
