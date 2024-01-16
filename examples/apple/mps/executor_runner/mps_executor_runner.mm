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
#include <iostream>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <executorch/sdk/bundled_program/bundled_program.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/extension/evalue_util/print_evalue.h>

#include <chrono>
using namespace std::chrono;

static constexpr size_t kRuntimeMemorySize = 4 * 1024U * 1024U; // 4 MB
static uint8_t runtime_pool[kRuntimeMemorySize];
static constexpr size_t kBundledAllocatorPoolSize = 16 * 1024U;
static uint8_t bundled_allocator_pool[kBundledAllocatorPoolSize];

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

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

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
        util::FileDataLoader::From(filename.c_str());
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
    Error status = torch::executor::bundled_program::GetProgramData(
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

struct TensorData {
  std::vector<uint8_t> data;
  ssize_t numel;
  size_t nbytes;
  ScalarType scalar_type;
};

template <typename T>
bool data_is_close_(
    const T* a,
    const T* b,
    size_t numel,
    double rtol,
    double atol) {
  for (size_t i = 0; i < numel; i++) {
    if (rtol == 0 && atol == 0) {
      // Exact comparison; avoid unnecessary math.
      if (a[i] != b[i]) {
        return false;
      }
    } else {
      auto allowed_error = atol + fabs(rtol * b[i]);
      auto actual_error = fabs(a[i] - b[i]);
      if (!isfinite(actual_error) || actual_error > allowed_error) {
        return false;
      }
    }
  }
  return true;
}

static
bool tensors_are_close_(
    ScalarType scalar_type,
    ssize_t numel,
    size_t nbytes,
    void* a_data_ptr,
    void* b_data_ptr,
    double rtol = 1e-05,
    double atol = 1e-08) {
  if (scalar_type == ScalarType::Float) {
    return data_is_close_<float>(
        (float*)a_data_ptr,
        (float*)b_data_ptr,
        numel,
        rtol,
        atol);
  } else if (scalar_type == ScalarType::Double) {
    return data_is_close_<double>(
        (double*)a_data_ptr,
        (double*)b_data_ptr,
        numel,
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(a_data_ptr, b_data_ptr, nbytes) == 0;
  }
}

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

  // Read in the entire file.
  Result<FreeableBuffer> file_data = loader->Load(0, loader->size().get());
  ET_CHECK_MSG(
      file_data.ok(),
      "Could not load contents of file '%s': 0x%x",
      model_path,
      (unsigned int)file_data.error());

  // Find the offset to the embedded Program.
  const void* program_data;
  size_t program_data_len;
  Error status = torch::executor::bundled_program::GetProgramData(
      const_cast<void*>(file_data->data()),
      file_data->size(),
      &program_data,
      &program_data_len);
  ET_CHECK_MSG(
      status == Error::Ok,
      "GetProgramData() failed on file '%s': 0x%x",
      model_path,
      (unsigned int)status);

  auto buffer_data_loader =
      util::BufferDataLoader(program_data, program_data_len);

  // Parse the program file. This is immutable, and can also be reused
  // between multiple execution invocations across multiple threads.
  Result<Program> program = Program::Load(&buffer_data_loader);
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const auto method_name_result = program->get_method_name(0);
  ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
  const char* method_name = *method_name_result;
  ET_LOG(Info, "Program methods: %zu", program->num_methods());

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
        Info, "Setting up non-const buffer %zu, size %lld.", id, *buffer_size);
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

  ET_LOG(
      Info, "Setting up memory manager");
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

  ET_LOG(
      Info, "Loading method name from plan");
  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      method.error());
  ET_LOG(Info, "Method loaded.");

  // Prepare the inputs.
  exec_aten::ArrayRef<void*> inputs;
  if (FLAGS_bundled_program) {
    ET_LOG(Info, "Loading bundled program...\n");
    // Use the inputs embedded in the bundled program.
    status = torch::executor::bundled_program::LoadBundledInput(
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
    inputs = torch::executor::util::PrepareInputTensors(*method);
  }
  ET_LOG(Info, "Inputs prepared.");

  for (int i = 0; i < FLAGS_num_runs; i++) {
    auto start_exec_time = high_resolution_clock::now();
    // Run the model.
    Error status = method->execute();
    auto end_exec_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_exec_time - start_exec_time);
    if (FLAGS_profile) {
      ET_LOG(Info, "[Run %d] Inference time: %lld milliseconds", i, duration.count());
    }
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        status);
  }
  ET_LOG(Info, "Model executed successfully.");

  auto output_list =
      runtime_allocator.allocateList<EValue>(method->outputs_size());
  status = method->get_outputs(output_list, method->outputs_size());
  ET_CHECK(status == Error::Ok);

  // Print the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  // Print the first and last 100 elements of long lists of scalars.
  std::cout << torch::executor::util::evalue_edge_items(100);
  for (int i = 0; i < outputs.size(); ++i) {
    std::cout << "Output " << i << ": " << outputs[i] << std::endl;
  }

  // Dump the profiling data to the specified file.
  torch::executor::prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen(FLAGS_prof_result_path.c_str(), "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  // Handle the outputs.
  if (FLAGS_bundled_program) {
    double rtol = 1e-05;
    double atol = 1e-08;

    if (strstr(model_path, "mv3")                  ||
        strstr(model_path, "mv2")                  ||
        strstr(model_path, "vit")                  ||
        strstr(model_path, "resnet18")             ||
        strstr(model_path, "resnet50")             ||
        strstr(model_path, "mobilebert")           ||
        strstr(model_path, "emformer")             ||
        strstr(model_path, "emformer_transcribe")  ||
        strstr(model_path, "emformer_join")        ||
        strstr(model_path, "edsr")                 ||
        strstr(model_path, "llama2")               ||
        strstr(model_path, "ic3")                  ||
        strstr(model_path, "ic4")) {
        atol = 1e-04;
    }
    status = torch::executor::bundled_program::VerifyResultWithBundledExpectedOutput(
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
  } else {
    util::FreeInputs(inputs);
  }
  return 0;
}
