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

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

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
#endif

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

static uint8_t temp_allocator_pool[1024U * 1024U];

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(data_path, "", "Path to data file.");
DEFINE_string(inputs, "", "Comma-separated list of input files");
DEFINE_string(
    output_file,
    "",
    "Base name of output file. If not empty output will be written to the file(s).");
DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");
DEFINE_string(
  output_folder_path,
  "outputs",
  "Executorch inference data output path.");

DEFINE_bool(dump_statistics, false, "Dump inference statistics.");
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
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy usecase between the application and device/co-processor associated with the backend.");
DEFINE_uint32(method_index, 0, "Index of methods to be specified.");

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::extension::FlatTensorDataMap;
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
#endif // ET_USE_THREADPOOL
  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  // Load .ptd file if provided
  std::unique_ptr<FileDataLoader> ptd_loader;
  std::unique_ptr<FlatTensorDataMap> ptd_data_map;
  if (!FLAGS_data_path.empty()) {
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

  std::vector<std::string> inputs_storage;
  std::vector<std::pair<char*, size_t>> input_buffers;

  std::stringstream list_of_input_files(FLAGS_inputs);
  std::string path;

  // First reserve memory for number of vector elements to avoid vector
  // reallocations when emplacing back.
  std::vector<std::string> file_paths;
  while (std::getline(list_of_input_files, path, ',')) {
    file_paths.push_back(std::move(path));
  }
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

    if (!input_file_handle.read(&inputs_storage.back()[0], file_size)) {
      ET_LOG(Error, "Failed to read input file: %s\n", file_path.c_str());
      return 1;
    }

    input_buffers.emplace_back(&inputs_storage.back()[0], file_size);
  }

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(FLAGS_method_index);
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
  auto before_load = std::chrono::high_resolution_clock::now();
  Result<Method> method = program->load_method(
      method_name,
      &memory_manager,
      tracer.get_event_tracer(),
      ptd_data_map.get());

  auto after_load = std::chrono::high_resolution_clock::now();
  double interval_load =
      std::chrono::duration_cast<std::chrono::microseconds>(
          after_load - before_load)
          .count() /
      1000.0;
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  // QCOM change
  std::ifstream input_list(FLAGS_input_list_path);
  if (input_list.is_open()) {
    auto inputs = executorch::extension::prepare_input_tensors(*method);
    ET_LOG(Debug, "Preparing inputs.");
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Debug, "Inputs prepared.");

    size_t num_inputs = method->inputs_size();
    ET_LOG(Info, "Number of inputs: %zu", num_inputs);

    auto split = [](std::string s, std::string delimiter) {
      size_t pos_start = 0, pos_end, delim_len = delimiter.length();
      std::string token;
      std::vector<std::string> res;

      while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
      }
      res.push_back(s.substr(pos_start));
      return res;
    };

    std::string file_path;
    int inference_index = 0;
    double elapsed_time = 0;
    while (std::getline(input_list, file_path)) {
      auto input_files = split(file_path, " ");
      if (input_files.size() == 0) {
        break;
      }
      ET_CHECK_MSG(
          input_files.size() == num_inputs,
          "Number of inputs (%zu) mismatch with input files (%zu)",
          num_inputs,
          input_files.size());

      std::vector<std::vector<char>> input_buf(num_inputs);
      for (int input_index = 0; input_index < num_inputs; ++input_index) {
        MethodMeta method_meta = method->method_meta();
        Result<executorch::runtime::TensorInfo> tensor_meta =
            method_meta.input_tensor_meta(input_index);

        std::ifstream fin(input_files[input_index], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();

        input_buf[input_index].resize(file_size);
        fin.seekg(0, fin.beg);
        fin.read(
            static_cast<char*>(input_buf[input_index].data()),
            file_size);
        fin.close();

        ET_CHECK_MSG(
            file_size == tensor_meta->nbytes(),
            "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
            input_index,
            file_size,
            tensor_meta->nbytes());

        auto impl = executorch::aten::TensorImpl(
            tensor_meta->scalar_type(),
            /*dim=*/tensor_meta->sizes().size(),
            const_cast<executorch::aten::TensorImpl::SizesType*>(tensor_meta->sizes().data()),
            input_buf[input_index].data(),
            const_cast<executorch::aten::TensorImpl::DimOrderType*>(
                tensor_meta->dim_order().data()));
        Error ret = method->set_input(executorch::aten::Tensor(&impl), input_index);
        ET_CHECK_MSG(
            ret == Error::Ok, "Failed to set input tensor: %d", (int)ret);
      }
      Error status = method->execute();
      std::vector<EValue> outputs(method->outputs_size());
      status = method->get_outputs(outputs.data(), method->outputs_size());
      ET_CHECK(status == Error::Ok);
      for (size_t output_index = 0; output_index < method->outputs_size();
          output_index++) {
        auto output_tensor = outputs[output_index].toTensor();
        size_t nbytes = output_tensor.nbytes();
        auto output_file_name = FLAGS_output_folder_path + "/output_" +
            std::to_string(inference_index) + "_" +
            std::to_string(output_index) + ".raw";
        std::ofstream fout(output_file_name.c_str(), std::ios::binary);
        fout.write(output_tensor.const_data_ptr<char>(), nbytes);
        fout.close();
      }
      ++inference_index;
    }
    return 0;
  } else {
    et_timestamp_t time_spent_executing = 0, time_spent_executing_1st = 0;
    auto inputs = executorch::extension::prepare_input_tensors(*method);
    ET_LOG(Info, "Preparing inputs.");
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Info, "Inputs prepared.");

    auto before_exec = std::chrono::high_resolution_clock::now();
    Error status = method->execute();
    auto after_exec = std::chrono::high_resolution_clock::now();
    double interval_1st_infs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            after_exec - before_exec)
            .count() /
        1000.0;

    before_exec = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < FLAGS_num_executions; i++) {
      status = method->execute();
      ET_CHECK_MSG(
          status == Error::Ok,
          "Execution of method %s failed with status 0x%" PRIx32,
          method_name,
          (uint32_t)status);
    }
    after_exec = std::chrono::high_resolution_clock::now();
    double interval_infs = std::chrono::duration_cast<std::chrono::microseconds>(
                              after_exec - before_exec)
                              .count() /
        1000.0 / FLAGS_num_executions;

    if (FLAGS_dump_statistics) {
      auto output_file_name = "statistics.txt";
      std::ofstream fout(output_file_name);
      fout << "load: " + std::to_string(interval_load)
          << "\n1st: " + std::to_string(interval_1st_infs)
          << "\navg: " + std::to_string(interval_infs) << std::endl;
      fout.close();
    }
    ET_LOG(Info, "Model executed successfully.");
    return 0;
  }
  // QCOM change end

  et_timestamp_t time_spent_executing = 0;
  // Run the model.
  for (uint32_t i = 0; i < FLAGS_num_executions; i++) {
    ET_LOG(Debug, "Preparing inputs.");
    // Allocate input tensors and set all of their elements to 1 or to the
    // contents of input_buffers if available. The `inputs`
    // variable owns the allocated memory and must live past the last call to
    // `execute()`.
    //
    // NOTE: we have to re-prepare input tensors on every execution
    // because inputs whose space gets reused by memory planning (if
    // any such inputs exist) will not be preserved for the next
    // execution.
    auto inputs = executorch::extension::prepare_input_tensors(
        *method, {}, input_buffers);
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Debug, "Inputs prepared.");

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
        (uint32_t)status);
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

  return 0;
}
