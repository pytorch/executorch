/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/sdk/etdump/etdump.h>
#include <executorch/bundled_program/runtime/bundled_program_verification.h>
#include <executorch/util/util.h>
#ifdef USE_ATEN_LIB
#include <c10/core/impl/LocalDispatchKeySet.h>
#endif

#if !defined(USE_ATEN_LIB)
#include <executorch/backends/xnnpack/threadpool/fb/threadpool_use_n_threads.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#endif

// This tool includes all of the headers necessary to execute a model.
// Demonstrate that those headers do not expose the internal flatbuffers
// headers.
#ifdef FLATBUFFERS_VERSION_MAJOR
// FLATBUFFERS_VERSION_MAJOR is defined by flatbuffers/base.h, which is included
// by all other flatbuffers library headers and by any generated headers. If
// it's present, it means that this file is including a flatbuffers header
// somewhere.
#error "The executorch headers must not expose flatbuffers.h"
#endif

using namespace torch::executor;

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
static constexpr size_t kBundledAllocatorPoolSize = 16 * 1024U;
static uint8_t bundled_allocator_pool[kBundledAllocatorPoolSize];

DEFINE_bool(
    bundled_program,
    false,
    "True for running bundled program, false for executorch_flatbuffer::program");

DEFINE_bool(
    generate_etdump,
    false,
    "If enabled etdump containing profiling data will be generated");

DEFINE_string(
    etdump_path,
    "etdump.etdp",
    "If etdump generation is enabled an etdump will be written out to this path");

DEFINE_string(
    prof_result_path,
    "prof_result.bin",
    "Executorch profiler output path.");

DEFINE_bool(print_output, false, "Prints output of the model.");

DEFINE_int32(num_iters, 1, "Number of inference iterations to run.");

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_int32(num_threads, 1, "Number of threads to use.");

DEFINE_string(
    method_name,
    "forward",
    "Name of method to run. Only used by bundled program mode.");

DEFINE_int32(
    testset_idx,
    0,
    "Index of bundled verification set to be run "
    "by bundled model for verification");

DEFINE_double(
    rtol,
    1e-5,
    "The relative tolerance used for bundled program verification.");

DEFINE_double(
    atol,
    1e-8,
    "The absolute tolerance used for bundled program verification.");

/**
 * Helps handle bundled and non-bundled program inputs.
 */
class ProgramData {
 public:
  /**
   * Tries loading the named file as a plain Program or a bundled program,
   * failing with an ET_CHECK on any failure.
   */
  static ProgramData load_or_die(std::string& filename) {
    // Create a DataLoader that wraps the input file. It may be a plain Program,
    // or it may be a BundledProgram that contains a Program.
    Result<util::FileDataLoader> loader =
        util::FileDataLoader::from(filename.c_str());
    ET_CHECK_MSG(
        loader.ok(),
        "Could not create loader for file '%s': 0x%x",
        filename.c_str(),
        (unsigned int)loader.error());

    // Figure out the file type. Create a scope to destroy the header after the
    // check.
    {
      Result<FreeableBuffer> header =
          loader->Load(/*offset=*/0, Program::kMinHeadBytes);
      ET_CHECK_MSG(
          header.ok(),
          "Could not load header of file '%s': 0x%x",
          filename.c_str(),
          (unsigned int)loader.error());
      Program::HeaderStatus hs =
          Program::check_header(header->data(), header->size());
      if (hs == Program::HeaderStatus::CompatibleVersion) {
        // It's a plain Program. We can use the existing loader, and there is no
        // bundled program data.
        return ProgramData(
            new util::FileDataLoader(std::move(*loader)),
            /*bundled_program_data=*/FreeableBuffer());
      }
    }

    // Try treating it as a bundled program.

    // Read in the entire file.
    Result<FreeableBuffer> file_data = loader->Load(0, loader->size().get());
    ET_CHECK_MSG(
        file_data.ok(),
        "Could not load contents of file '%s': 0x%x",
        filename.c_str(),
        (unsigned int)file_data.error());

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
        filename.c_str(),
        (unsigned int)status);

    // Wrap the Program in a loader, and pass on the FreeableBuffer that
    // contains the full bundled program data.
    return ProgramData(
        new util::BufferDataLoader(program_data, program_data_len),
        std::move(*file_data));
  }

  /**
   * Returns the loader for the plain Program. May or may not be inside a
   * bundled program wrapper.
   */
  DataLoader* program_loader() {
    return loader_.get();
  }

  /**
   * If the file was a bundled program, returns a pointer to the file data.
   * Otherwise returns nullptr.
   */
  const void* bundled_program_data() const {
    if (bundled_program_data_.size() > 0) {
      return bundled_program_data_.data();
    } else {
      return nullptr;
    }
  }

 private:
  /// Takes ownership of both params.
  ProgramData(DataLoader* loader, FreeableBuffer&& bundled_program_data)
      : loader_(loader),
        bundled_program_data_(std::move(bundled_program_data)) {}

  std::unique_ptr<DataLoader> loader_;
  FreeableBuffer bundled_program_data_;
};

int main(int argc, char** argv) {
  torch::executor::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args: ";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  ET_CHECK_MSG(
      FLAGS_num_threads >= 1,
      "Please specifiy valid number of threads to use.");

  // Load the file.
  auto program_data = ProgramData::load_or_die(FLAGS_model_path);

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");
  Result<Program> program =
      torch::executor::Program::load(program_data.program_loader());
  EXECUTORCH_END_PROF(prof_tok);
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", FLAGS_model_path.c_str());
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", FLAGS_model_path.c_str());

  // Use the first method in the program.
  const size_t method_index = 0;
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(method_index);
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

  prof_tok = EXECUTORCH_BEGIN_PROF("load model");
  Result<Method> method = program->load_method(method_name, &memory_manager);
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK_MSG(
      method.ok(),
      "load_method() failed with status 0x%" PRIx32,
      method.error());

  ET_LOG(Info, "Model initialized.");

#ifdef USE_ATEN_LIB
  // [TLS handling] This is to workaround an assertion failure
  // (https://fburl.com/code/302jyn8d) running `gelu` in ATen mode in fbcode
  // (such as bento). The problem is Executorch ATen mode doesn't have Thread
  // Local State, but `torch-cpp` is assuming tls init is done. There are two
  // more checks: MKLDNN disabled and C10_MOBILE, if any of them is true we
  // won't be hitting this assertion error. However in `torch-cpp` lib both
  // checks are false. Production impact: this should not make any impact in
  // production environment, given that in xplat we are depending on a library
  // that enables C10_MOBILE (`torch_mobile_core`).
  c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
#endif

#if !defined(USE_ATEN_LIB)
  // To enable intra-op parallelism
  // This sets the # of threads to use for running executorch model
  // to num_threads. Applicable to lean mode.
  torch::executorch::threadpool::UseNThreadsThreadPoolGuard thread_pool_guard(
      FLAGS_num_threads);
  ET_CHECK_MSG(
      thread_pool_guard.guard_armed(),
      "Could not set # of threads to use. "
      "Num threads requested is %d, Threadpool size is: %ld",
      FLAGS_num_threads,
      torch::executorch::threadpool::get_threadpool()->get_thread_count());
#endif
  // Run the model multiple times if requested.
  MemoryAllocator bundled_input_allocator{
      MemoryAllocator(kBundledAllocatorPoolSize, bundled_allocator_pool)};
  Error status;
  for (size_t i = 0; i < FLAGS_num_iters; i++) {
    // Prepare the inputs.
    exec_aten::ArrayRef<void*> inputs;
    if (FLAGS_bundled_program) {
      // Use the inputs embedded in the bundled program.
      status = torch::executor::util::LoadBundledInput(
          *method,
          program_data.bundled_program_data(),
          &bundled_input_allocator,
          method_name,
          FLAGS_testset_idx);
      ET_CHECK_MSG(
          status == Error::Ok,
          "LoadBundledInput failed with status 0x%" PRIx32,
          status);
    } else {
      // Use ones-initialized inputs.
      inputs = torch::executor::util::PrepareInputTensors(*method);
    }
    ET_LOG(Info, "Inputs prepared.");

    // Run the model.
    EXECUTORCH_PROFILE_CREATE_BLOCK("inference loop");
    prof_tok = EXECUTORCH_BEGIN_PROF("run model");
    status = method->execute();
    EXECUTORCH_END_PROF(prof_tok);
    ET_CHECK_MSG(
        status == Error::Ok,
        "method->execute() failed with status 0x%" PRIx32,
        status);
    ET_LOG(Info, "Model executed successfully.");

    // Handle the outputs.
    if (FLAGS_bundled_program) {
      status = torch::executor::util::VerifyResultWithBundledExpectedOutput(
          *method,
          program_data.bundled_program_data(),
          &bundled_input_allocator,
          method_name,
          FLAGS_testset_idx,
          FLAGS_rtol,
          FLAGS_atol);
      ET_CHECK_MSG(
          status == Error::Ok,
          "Bundle verification failed with status 0x%" PRIx32,
          status);
      ET_LOG(Info, "Model verified successfully.");
    } else {
      torch::executor::util::FreeInputs(inputs);
    }
  }

  // Print the outputs if requested.
  if (FLAGS_print_output) {
    auto output_list = std::make_unique<EValue[]>(method->outputs_size());
    status = method->get_outputs(output_list.get(), method->outputs_size());
    ET_CHECK_MSG(
        status == Error::Ok,
        "get_outputs failed with status 0x%" PRIx32,
        status);

    // TODO(T139071931): Don't assume that all outputs are tensors.
    for (size_t i = 0; i < method->outputs_size(); i++) {
      auto output_tensor = output_list[i].toTensor();
      const float* data_output = output_tensor.const_data_ptr<float>();
      for (size_t j = 0; j < output_list[i].toTensor().numel(); ++j) {
        ET_LOG(Info, "%f", data_output[j]);
      }
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

  if (FLAGS_generate_etdump) {
    ETDump et_dump(method_allocator);
    auto ret =
        et_dump.serialize_prof_results_to_etdump(FLAGS_etdump_path.c_str());
    if (ret != torch::executor::Error::Ok) {
      ET_LOG(Error, "Failed to serialize and write out etdump data.");
      return -1;
    }
  }

  return 0;
}
