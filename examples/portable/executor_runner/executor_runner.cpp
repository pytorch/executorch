/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2024-2025 Arm Limited and/or its affiliates.
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

#include <cstdint>
#ifdef ET_BUNDLE_IO_ENABLED
#include <filesystem>
#endif // ET_BUNDLE_IO_ENABLED
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif // ET_EVENT_TRACER_ENABLED

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/extension/threadpool/threadpool_guard.h>
#endif

#ifdef ET_BUNDLE_IO_ENABLED
#include <executorch/devtools/bundled_program/bundled_program.h>
#endif

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

static uint8_t temp_allocator_pool[1024U * 1024U];

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(data_path, "", "Path to data file (.ptd).");
DEFINE_string(inputs, "", "Comma-separated list of input files");
DEFINE_string(
    output_file,
    "",
    "Base name of output file. If not empty output will be written to the file(s).");

DEFINE_bool(
    print_all_output,
    false,
    "Prints all output. By default only first and last 100 elements are printed.");
DEFINE_uint32(num_executions, 1, "Number of times to run the model.");
#ifdef ET_EVENT_TRACER_ENABLED
DEFINE_string(etdump_path, "model.etdump", "Write ETDump data to this path.");
#endif // ET_EVENT_TRACER_ENABLED
DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

#ifdef ET_BUNDLE_IO_ENABLED
DEFINE_double(bundleio_rtol, 0.01, "Relative tolerance for bundled IO.");
DEFINE_double(bundleio_atol, 0.01, "Absolute tolerance for bundled IO.");
#endif

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
#ifdef ET_BUNDLE_IO_ENABLED
using executorch::bundled_program::compute_method_output_error_stats;
using executorch::bundled_program::ErrorStats;
using executorch::bundled_program::verify_method_outputs;
#endif
using executorch::extension::BufferDataLoader;
using executorch::extension::FileDataLoader;
using executorch::extension::FlatTensorDataMap;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::EventTracer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

/// Helper to manage resources for ETDump generation
class EventTraceManager {
 public:
  EventTraceManager() : event_tracer_ptr_(nullptr) {
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_ptr_ = std::make_shared<executorch::etdump::ETDumpGen>();
#endif // ET_EVENT_TRACER_ENABLED
  }

  EventTracer* get_event_tracer() const {
    return event_tracer_ptr_.get();
  };

  Error write_etdump_to_file() const {
    EventTracer* const event_tracer_ptr = get_event_tracer();
    if (!event_tracer_ptr) {
      return Error::NotSupported;
    }

#ifdef ET_EVENT_TRACER_ENABLED
    executorch::etdump::ETDumpGen* const etdump_ptr =
        static_cast<executorch::etdump::ETDumpGen*>(event_tracer_ptr);

    const char* filename = FLAGS_etdump_path.c_str();

    std::unique_ptr<FILE, decltype(&fclose)> etdump_file(
        fopen(filename, "w+"), fclose);
    if (!etdump_file) {
      ET_LOG(Error, "Failed to open ETDump file at %s.", filename);
      return Error::AccessFailed;
    }

    executorch::etdump::ETDumpResult result = etdump_ptr->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      fwrite((uint8_t*)result.buf, 1, result.size, etdump_file.get());
      free(result.buf);
      ET_LOG(Info, "ETDump written to file '%s'.", filename);
    } else {
      ET_LOG(Error, "No ETDump data available!");
      return Error::NotFound;
    }
#endif // ET_EVENT_TRACER_ENABLED

    return Error::Ok;
  }

 private:
  std::shared_ptr<EventTracer> event_tracer_ptr_;
};

#ifdef ET_BUNDLE_IO_ENABLED
std::vector<uint8_t> try_load_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(
      file.is_open(), "Could not open file '%s'", path.string().c_str());

  const std::size_t nbytes = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> file_data(nbytes);
  ET_CHECK_MSG(
      file.read(reinterpret_cast<char*>(file_data.data()), nbytes),
      "Could not load contents of file '%s'",
      path.string().c_str());

  return file_data;
}
#endif

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

#if defined(ET_USE_THREADPOOL)
  auto cpu_threads = FLAGS_cpu_threads;
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
  std::optional<::executorch::extension::threadpool::NoThreadPoolGuard>
      opt_guard;
  if (cpu_threads == 1) {
    opt_guard.emplace();
  }
#endif // ET_USE_THREADPOOL

  bool bundle_io = false;
  size_t program_data_len = 0;
  const void* program_data = nullptr;

#ifdef ET_BUNDLE_IO_ENABLED
  std::vector<uint8_t> model_file_data = try_load_file(FLAGS_model_path);
  uint8_t* model_pte = model_file_data.data();
  size_t pte_size = model_file_data.size();
  constexpr size_t testset_idx = 0;

  // Check for bundled IO provided model.
  bundle_io = executorch::bundled_program::is_bundled_program(
      reinterpret_cast<void*>(model_pte), pte_size);

  if (bundle_io) {
    // BundleIO bpte file is provided - dig out the actual model from the data
    // area.
    ET_LOG(Debug, "PTE Model with bundle io detected.");
    Error status = executorch::bundled_program::get_program_data(
        reinterpret_cast<void*>(model_pte),
        pte_size,
        &program_data,
        &program_data_len);

    ET_CHECK_MSG(
        status == Error::Ok,
        "get_program_data() from bundle PTE failed: 0x%x" PRIx32,
        static_cast<uint32_t>(status));
  } else {
    ET_LOG(Debug, "PTE Model has no bundled IO");
  }
#endif

  // Inputs can come from bundleio, as optional input file(s), or
  // everything hardcoded to ones.
  std::vector<std::string> inputs_storage;
  std::vector<std::pair<char*, size_t>> input_buffers;
  if (!bundle_io) {
    if (!FLAGS_inputs.empty()) {
      ET_LOG(Info, "Loading inputs from input file(s).");
      std::stringstream list_of_input_files(FLAGS_inputs);
      std::string path;

      std::vector<std::string> file_paths;
      while (std::getline(list_of_input_files, path, ',')) {
        file_paths.push_back(std::move(path));
      }
      // First reserve number of elements to avoid vector reallocations.
      inputs_storage.reserve(file_paths.size());

      for (const auto& file_path : file_paths) {
        std::ifstream input_file_handle(
            file_path, std::ios::binary | std::ios::ate);

        if (!input_file_handle) {
          ET_LOG(Error, "Failed to open input file: %s\n", file_path.c_str());
          return 1;
        }

        std::streamsize file_size = input_file_handle.tellg();
        input_file_handle.seekg(0, std::ios::beg);

        // Reserve memory for actual file contents.
        inputs_storage.emplace_back(file_size, '\0');

        if (!input_file_handle.read(inputs_storage.back().data(), file_size)) {
          ET_LOG(Error, "Failed to read input file: %s\n", file_path.c_str());
          return 1;
        }

        input_buffers.emplace_back(&inputs_storage.back()[0], file_size);
      }
    }
  }

  std::unique_ptr<FileDataLoader> ptd_loader;
  std::unique_ptr<FlatTensorDataMap> ptd_data_map;
  if (!FLAGS_data_path.empty()) {
    ET_LOG(Info, "Loading tensor data from .ptd file.");
    const char* data_path = FLAGS_data_path.c_str();
    Result<FileDataLoader> ptd_loader_result = FileDataLoader::from(data_path);
    ET_CHECK_MSG(
        ptd_loader_result.ok(),
        "FileDataLoader::from() failed for PTD file: 0x%" PRIx32,
        (uint32_t)ptd_loader_result.error());
    ptd_loader =
        std::make_unique<FileDataLoader>(std::move(ptd_loader_result.get()));
    ET_LOG(Info, "PTD file %s is loaded.", data_path);

    Result<FlatTensorDataMap> ptd_data_map_result =
        FlatTensorDataMap::load(ptd_loader.get());
    ET_CHECK_MSG(
        ptd_data_map_result.ok(),
        "FlatTensorDataMap::load() failed for PTD file: 0x%" PRIx32,
        (uint32_t)ptd_data_map_result.error());
    ptd_data_map = std::make_unique<FlatTensorDataMap>(
        std::move(ptd_data_map_result.get()));
    ET_LOG(
        Info,
        "PTD data map created with %" PRIu64 " keys.",
        static_cast<uint64_t>(ptd_data_map->get_num_keys().get()));
  }

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  std::unique_ptr<DataLoader> loader;

  if (bundle_io) {
    Result<BufferDataLoader> buffer_loader =
        BufferDataLoader(program_data, program_data_len);
    ET_CHECK_MSG(
        buffer_loader.ok(),
        "BufferDataLoader failed: 0x%" PRIx32,
        static_cast<uint32_t>(buffer_loader.error()));
    ET_LOG(
        Debug,
        "Bundled IO PTE Model data loaded. Size: %zu bytes.",
        program_data_len);
    loader = std::make_unique<BufferDataLoader>(std::move(buffer_loader.get()));
  } else {
    Result<FileDataLoader> file_loader =
        FileDataLoader::from(FLAGS_model_path.c_str());
    ET_CHECK_MSG(
        file_loader.ok(),
        "FileDataLoader::from() failed: 0x%" PRIx32,
        static_cast<uint32_t>(file_loader.error()));
    loader = std::make_unique<FileDataLoader>(std::move(file_loader.get()));
  }

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", FLAGS_model_path.c_str());
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", FLAGS_model_path.c_str());

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

  // Temporary memory required by kernels
  MemoryAllocator temp_allocator{
      MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

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
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  //
  // Load the method from the program, using the provided allocators. Running
  // the method can mutate the memory-planned buffers, so the method should only
  // be used by a single thread at at time, but it can be reused.
  //
  EventTraceManager tracer;
  Result<Method> method = program->load_method(
      method_name,
      &memory_manager,
      tracer.get_event_tracer(),
      ptd_data_map.get());
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  et_timestamp_t time_spent_executing = 0;
  // Run the model.
  for (uint32_t i = 0; i < FLAGS_num_executions; i++) {
    // Allocate input tensors and set all of their elements to 1 or to the
    // contents of input_buffers if available. For non bundled IO, the `inputs`
    // variable owns the allocated memory and must live past the last call to
    // `execute()`.
    //
    // NOTE: we have to re-prepare input tensors on every execution
    // because inputs whose space gets reused by memory planning (if
    // any such inputs exist) will not be preserved for the next
    // execution.
    std::optional<executorch::extension::BufferCleanup> inputs;

#ifdef ET_BUNDLE_IO_ENABLED
    if (bundle_io) {
      ET_LOG(Debug, "Getting inputs from bundled IO");
      Error status = executorch::bundled_program::load_bundled_input(
          *method, model_pte, testset_idx);
      ET_CHECK_MSG(
          status == Error::Ok,
          "load_bundled_input failed with status 0x%" PRIx32,
          static_cast<uint32_t>(status));
    } else
#endif
    {
      ET_LOG(Debug, "Preparing inputs.");
      auto res = executorch::extension::prepare_input_tensors(
          *method, {}, input_buffers);
      ET_CHECK_MSG(
          res.ok(),
          "Could not prepare inputs: 0x%" PRIx32,
          (uint32_t)res.error());
      inputs.emplace(std::move(res.get()));
      ET_LOG(Debug, "Inputs prepared.");
    }

    const et_timestamp_t before_execute =
        executorch::runtime::pal_current_ticks();
    Error status = method->execute();
    const et_timestamp_t after_execute =
        executorch::runtime::pal_current_ticks();
    time_spent_executing += after_execute - before_execute;
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        static_cast<uint32_t>(status));
  }
  const auto tick_ratio = et_pal_ticks_to_ns_multiplier();
  constexpr auto NANOSECONDS_PER_MILLISECOND = 1000000;
  ET_LOG(
      Info,
      "Model executed successfully %" PRIu32 " time(s) in %f ms.",
      FLAGS_num_executions,
      static_cast<double>(time_spent_executing) * tick_ratio.numerator /
          tick_ratio.denominator / NANOSECONDS_PER_MILLISECOND);

  // Print the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  Error status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  if (FLAGS_output_file.size() > 0) {
    for (int i = 0; i < outputs.size(); ++i) {
      if (outputs[i].isTensor()) {
        Tensor tensor = outputs[i].toTensor();

        char out_filename[255];
        snprintf(out_filename, 255, "%s-%d.bin", FLAGS_output_file.c_str(), i);
        ET_LOG(Info, "Writing output to file: %s", out_filename);
        FILE* out_file = fopen(out_filename, "wb");
        fwrite(tensor.const_data_ptr<char>(), 1, tensor.nbytes(), out_file);
        fclose(out_file);
      }
    }
  }

  if (FLAGS_print_all_output) {
    for (int i = 0; i < outputs.size(); ++i) {
      if (outputs[i].isTensor()) {
        Tensor tensor = outputs[i].toTensor();

        for (int j = 0; j < tensor.numel(); ++j) {
          if (tensor.scalar_type() == ScalarType::Int) {
            printf(
                "Output[%d][%d]: (int) %d\n",
                i,
                j,
                tensor.const_data_ptr<int>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Float) {
            printf(
                "Output[%d][%d]: (float) %f\n",
                i,
                j,
                tensor.const_data_ptr<float>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Char) {
            printf(
                "Output[%d][%d]: (char) %d\n",
                i,
                j,
                tensor.const_data_ptr<int8_t>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Bool) {
            printf(
                "Output[%d][%d]: (bool) %s (0x%x)\n",
                i,
                j,
                tensor.const_data_ptr<int8_t>()[j] ? "true " : "false",
                tensor.const_data_ptr<int8_t>()[j]);
          }
        }
      } else {
        printf("Output[%d]: Not Tensor\n", i);
      }
    }
  } else {
    // Print the first and last 100 elements of long lists of scalars.
    std::cout << executorch::extension::evalue_edge_items(100);

    for (int i = 0; i < outputs.size(); ++i) {
      std::cout << "OutputX " << i << ": " << outputs[i] << std::endl;
    }
  }

  if (tracer.get_event_tracer()) {
    // Dump ETDump data containing profiling/debugging data to file specified in
    // command line flag.
    status = tracer.write_etdump_to_file();
    ET_CHECK_MSG(status == Error::Ok, "Failed to save ETDump file.");
  }

#ifdef ET_BUNDLE_IO_ENABLED
  if (bundle_io) {
    // With bundled io we can check the result.
    bool model_ok = false;

    ErrorStats stats =
        compute_method_output_error_stats(*method, model_pte, testset_idx);

    if (stats.status == Error::Ok) {
      ET_LOG(Info, "=== Error stats for testset %zu ===", testset_idx);
      ET_LOG(Info, " mean_absolute_error: %f", stats.mean_abs_error);
      ET_LOG(Info, " max_absolute_error:  %f", stats.max_abs_error);
      ET_LOG(Info, " mean_relative_error: %f", stats.mean_relative_error);
      ET_LOG(Info, " max_relative_error:  %f", stats.max_relative_error);
    } else {
      ET_LOG(
          Info,
          "=== Error calculating stats for testset %zu ERROR: 0x%x" PRIx32
          "===",
          testset_idx,
          static_cast<uint32_t>(stats.status));
    }

    Error status = verify_method_outputs(
        *method,
        model_pte,
        testset_idx,
        FLAGS_bundleio_rtol,
        FLAGS_bundleio_atol);
    if (status == Error::Ok) {
      ET_LOG(Info, "Model output match expected BundleIO bpte ref data.");
      ET_LOG(Info, "TEST: BundleIO index[%zu] Test_result: PASS", testset_idx);
      model_ok = true;
    } else {
      ET_LOG(
          Error,
          "Model output don't match expected BundleIO bpte ref data. rtol=%f atol=%f",
          FLAGS_bundleio_rtol,
          FLAGS_bundleio_atol);
      ET_LOG(Error, "TEST: BundleIO index[%zu] Test_result: FAIL", testset_idx);
      ET_LOG(
          Error,
          "Bundle verification failed with status 0x%" PRIx32,
          static_cast<uint32_t>(status));
      model_ok = false;
    }

    if (!model_ok) {
      return 1;
    }
  }
#endif

  return 0;
}
