//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

/**
 * @file
 *
 * This tool can run Executorch model files that use operators that
 * are covered by the MPSDelegate or the portable kernels.
 *
 * It uses the original bundled input data from the flatbuffer file.
 */

#include <memory>
#include <numeric>
#include <iomanip>
#include <iostream>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>

#include <chrono>
using namespace std::chrono;

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(model_path, "model.ff", "Model serialized in flatbuffer format.");
DEFINE_string(
    prof_result_path,
    "prof_result.bin",
    "Executorch profiler output path.");

DEFINE_bool(
    bundled_program,
    false,
    "True for running bundled program, false for executorch_flatbuffer::program");

DEFINE_int32(
    testset_idx,
    0,
    "Index of bundled verification set to be run "
    "by bundled model for verification");

DEFINE_int32(
    num_runs,
    1,
    "Number of total runs");

DEFINE_bool(
    profile,
    false,
    "True for showing profile data (e.g execution time)");

DEFINE_bool(
    skip_warmup,
    false,
    "If true, a warmup iteration won't be executed.");

DEFINE_string(
    etdump_path,
    "etdump.etdp",
    "If etdump generation is enabled an etdump will be written out to this path");

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
using executorch::extension::BufferCleanup;
using executorch::extension::BufferDataLoader;
using executorch::extension::FileDataLoader;
using executorch::runtime::DataLoader;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace bundled_program = executorch::bundled_program;

int main(int argc, char** argv) {
  {
    const char* usage = R"(MPS Executor Runner. Sample usage:
  mps_executor_runner --model_path model.pte)";
    gflags::SetUsageMessage(usage);
  }

  if (argc == 1) {
    ET_LOG(Error, "No options provided.");
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

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

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::from() failed: 0x%" PRIx32, loader.error());

  // Read in the entire file.
  Result<FreeableBuffer> file_data = loader->load(0, loader->size().get(), DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ET_CHECK_MSG(
      file_data.ok(),
      "Could not load contents of file '%s': 0x%x",
      model_path,
      (unsigned int)file_data.error());

  // Find the offset to the embedded Program.
  const void* program_data;
  size_t program_data_len;
  Error status = bundled_program::get_program_data(
      const_cast<void*>(file_data->data()),
      file_data->size(),
      &program_data,
      &program_data_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "get_program_data() failed on file '%s': 0x%x",
      model_path,
      (unsigned int)status);

  // Wrap the buffer in a DataLoader.
  auto buffer_data_loader =
      BufferDataLoader(program_data, program_data_len);

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&buffer_data_loader);
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

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
      (uint32_t)method.error());
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

  // Prepare the inputs.
  std::unique_ptr<BufferCleanup> inputs;
  if (FLAGS_bundled_program) {
    ET_LOG(Info, "Loading bundled program...");
    // Use the inputs embedded in the bundled program.
    status = bundled_program::load_bundled_input(
        *method,
        file_data->data(),
        FLAGS_testset_idx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "LoadBundledInput failed with status 0x%" PRIx32,
        status);
  } else {
    ET_LOG(Info, "Loading non-bundled program...\n");
    // Use ones-initialized inputs.
    auto inputs_result = executorch::extension::prepare_input_tensors(*method);
    if (inputs_result.ok()) {
      // Will free the inputs when destroyed.
      inputs =
          std::make_unique<BufferCleanup>(std::move(inputs_result.get()));
    }
  }
  ET_LOG(Info, "Inputs prepared.");

  int num_iterations = FLAGS_num_runs + (FLAGS_skip_warmup ? 0 : 1);
  std::vector<float> exec_times;
  exec_times.reserve(FLAGS_num_runs);
  for (int i = 0; i < num_iterations; i++) {
    auto start_exec_time = high_resolution_clock::now();
    // Run the model.
    Error status = method->execute();
    auto end_exec_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_exec_time - start_exec_time);
    exec_times.push_back(duration.count());
    if (FLAGS_profile) {
      const float miliseconds = static_cast<float>(duration.count()) / 1000.f;
      ET_LOG(Info, "[Run %d] Inference time: %.3f miliseconds", i, miliseconds);
    }
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        status);
  }
  if (FLAGS_profile && FLAGS_num_runs) {
    auto itr = exec_times.begin();
    if (!FLAGS_skip_warmup)
      itr++;

    const float avg_time = (std::reduce(itr, exec_times.end()) / static_cast<float>(FLAGS_num_runs)) / 1000.f;
    std::cout << "Average inference time: " << std::setprecision(2) << std::fixed << avg_time << " miliseconds\n";
  }
  ET_LOG(Info, "Model executed successfully.");

  // Print the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  // Print the first and last 100 elements of long lists of scalars.
  std::cout << executorch::extension::evalue_edge_items(100);
  for (int i = 0; i < outputs.size(); ++i) {
    std::cout << "Output " << i << ": " << outputs[i] << std::endl;
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

  // Handle the outputs.
  if (FLAGS_bundled_program) {
    double rtol = 1e-05;
    double atol = 1e-08;
    if (strstr(model_path, "fp16")) {
      rtol = 1e-01;
      atol = 1e-01;
    } else if (strstr(model_path, "mv3")           ||
        strstr(model_path, "mv2")                  ||
        strstr(model_path, "conv")                 ||
        strstr(model_path, "vit")                  ||
        strstr(model_path, "resnet18")             ||
        strstr(model_path, "resnet50")             ||
        strstr(model_path, "emformer")             ||
        strstr(model_path, "emformer_transcribe")  ||
        strstr(model_path, "emformer_join")        ||
        strstr(model_path, "edsr")                 ||
        strstr(model_path, "llama2")               ||
        strstr(model_path, "ic3")                  ||
        strstr(model_path, "ic4")) {
      atol = 1e-04;
    } else if (strstr(model_path, "mobilebert")) {
      atol = 1e-01;
      rtol = 1e-01;
    }
    status = bundled_program::verify_method_outputs(
        *method,
        file_data->data(),
        FLAGS_testset_idx,
        rtol,
        atol
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
