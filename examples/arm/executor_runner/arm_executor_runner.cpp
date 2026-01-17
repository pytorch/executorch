/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* This is an example ExecuTorch runner running on Arm Cortex-M and Ethos-U
 * based hardware. This example tries to illustrate a few ways to use ExecuTorch
 * and you can use it as is or remove the unneeded parts. Please use this code
 * as inspiration.
 *
 * Some defines used to configure the code:
 *
 * ET_MODEL_PTE_ADDR  - Where in memory/flash your PTE model data is, if
 *                      not set the model is supposed to have been converted to
 *                      a c-array named model_pte and put into model_pte.h
 *                      this is placed in network_model_sec linker section
 *                      that is controlled by your memory mode via the
 *                      ETHOSU_MODEL cmake parameter.
 *                      If SEMIHOSTING is define this is not used
 * ET_NUM_INFERENCES  - Numbers of times to run the inference
 * ET_LOG_DUMP_INPUT  - Control if you want input to be dumped to the log.
 * ET_LOG_DUMP_OUTPUT     - Control if you want output to be dumped to the log.
 *
 * Devtool BundleIO: Use Bundle PTE with input and reference output included to
 * check if it matches.
 *
 * ET_BUNDLE_IO       - Build in Devtools BundleIO, this makes it possible to
 *                      use bpte with bundled input and output refdata to
 *                      compare output.
 *                      See also ET_ATOL and ET_RTOL
 *   ET_ATOL              - The atol used to compare the output and ref data
 * when using ET_BUNDLE_IO ET_RTOL              - The rtol used to compare the
 * output and ref data when using ET_BUNDLE_IO
 *
 * Devtools ETDump: Speed and dumping output
 *
 * ET_EVENT_TRACER_ENABLED       - Build in Devtools ETDump event trace code
 *                                 to generate cycle data and print it base64
 *                                 coded in the log so you can get it out of
 *                                 your embedded target. This can be used to
 *                                 benchmark where time is spent. If you run
 *                                 on Ethos-U the delegate/commandstream is
 *                                 run in one go, this means that per op
 *                                 measurements is not possible.
 *  ET_DUMP_OUTPUTS              - Collect and print outputs as a base64 buffer
 *                                 in the log, see ExecuTorch Devtools for more
 *                                 info. (Requires ET_EVENT_TRACER_ENABLED)
 *  ET_DUMP_INTERMEDIATE_OUTPUTS - Collect and print intermediate outputs as a
 *                                 base64 buffer in the log, see ExecuTorch
 *                                 Devtools for more info.
 *                                 (Requires ET_EVENT_TRACER_ENABLED)
 *  ET_DEBUG_BUFFER_SIZE         - Override the size of memory area used by
 *                                 ET_DUMP_OUTPUTS or
 * ET_DUMP_INTERMEDIATE_OUTPUTS
 *
 * Warning: CPU time measurements is NOT possible in the FVP simulator and a
 * real target or FPGA must be used. NPU number are roughly OK, and can be used
 * as guidance if timeing adaptor values are set correctly.
 *
 * SEMIHOSTING - When using the FVP simulator it can be built to access your dev
 *               machines filesystem, this is used for testing models in
 *               unittest/pytest and a special version of the runner is built
 *               to read model and input as files and output is saved to the
 *               filesystem. The backends/arm/test/setup_testing.sh script will
 *               build this for you so you can use it from pytest to test with
 *               the FVP simulator.
 *
 * Memory areas used:
 *    You might want to configure this differently on your HW, like maybe all
 *    left over memory after code is linked. This needs to be big enough to fit
 *    and run your model. In our example using the FVP simulator we have much
 *    memory and set this quite high to be able to test larger models.
 *    Regarding heap/mallocs type of allocation from ExecuTorch,
 *    et_pal_allocate() is not implemented or needed.
 *
 * ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE            - Size of memory area
 *                                                          used when setting up
 *                                                          the model
 * ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE - Size of memory area
 *                                                          used when running
 *                                                          inferences
 */

#include <errno.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "arm_memory_allocator.h"
#include "arm_perf_monitor.h"

#if defined(ET_BUNDLE_IO)
#include <executorch/devtools/bundled_program/bundled_program.h>
#endif

#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/devtools/etdump/etdump_flatcc.h>

#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
#include <executorch/devtools/etdump/data_sinks/buffer_data_sink.h>

#if !defined(ET_DEBUG_BUFFER_SIZE)
#define ET_DEBUG_BUFFER_SIZE (2 * 1024 * 1024)
#endif

#endif

#if !defined(SEMIHOSTING)
#include <executorch/third-party/flatcc/include/flatcc/portable/pbase64.h>
#endif

#endif // defined(ET_EVENT_TRACER_ENABLED)

#if defined(SEMIHOSTING)

/**
 * The input_file_allocation_pool should be large enough to fit the various
 * input file data used when loading the data files when running semihosting
 * e.g. the input file data and the pte file data
 * In our unit test flow, we have the capability to provide an enitre model to
 * the Corstone-3xx FVP using semi hosting. Hence, the input file allocation
 * pool needs to be large enough to take an entire model and input. On the FVP,
 * input_data_sec is linked to the DDR, which is large (256MB on
 * Corstone-300).
 * If you use semihosting on your HW this can be lowered to fit your
 * files/memory
 */

const size_t input_file_allocation_pool_size = 60 * 1024 * 1024;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) input_file_allocation_pool[input_file_allocation_pool_size];
char* model_pte = nullptr;

#else
#if defined(ET_MODEL_PTE_ADDR)

/**
 * Set ET_MODEL_PTE_ADDR to the memory address where your PTE is placed
 * e.g. if you for example flash it to 0x7000000 set
 * -DET_MODEL_PTE_ADDR=0x7000000 You can run the Corstone FVP with the --data
 * flag to place it on a address if you use the FVP.
 */
char* model_pte = reinterpret_cast<char*>(ET_MODEL_PTE_ADDR);

#else
/**
 * This header file is generated by the build process based on the .pte file
 * specified in the ET_PTE_FILE_PATH variable to the cmake build.
 * Control of the action of the .pte, it's use of operators and delegates, and
 * which are included in the bare metal build are also orchestrated by the
 * CMakeLists file. For example use see examples/arm/run.sh
 *
 * e.g. This includes the pte as a big chunk of data struct into this file
 */
#include "model_pte.h"
#endif
#endif

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
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
using executorch::runtime::toString;
#if defined(ET_BUNDLE_IO)
using executorch::bundled_program::compute_method_output_error_stats;
using executorch::bundled_program::ErrorStats;
using executorch::bundled_program::verify_method_outputs;
#endif
#if defined(ET_EVENT_TRACER_ENABLED)
using executorch::etdump::BufferDataSink;
using executorch::etdump::ETDumpGen;
using executorch::etdump::ETDumpResult;
using executorch::runtime::EventTracerDebugLogLevel;
using torch::executor::etdump_result;
#endif
/**
 * The method_allocation_pool should be large enough to fit the setup, input
 * used and other data used like the planned memory pool (e.g. memory-planned
 * buffers to use for mutable tensor data) In this example we run on a
 * Corstone-3xx FVP so we can use a lot of memory to be able to run and test
 * large models if you run on HW this should be lowered to fit into your
 * availible memory.
 */
#if !defined(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE (60 * 1024 * 1024)
#endif
const size_t method_allocation_pool_size =
    ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

#if defined(ET_BUNDLE_IO)

const size_t testset_idx = 0; // BundleIO test indexes to test if used

#if defined(ET_ATOL)
const float et_atol = ET_ATOL;
#else
const float et_atol = 0.01;
#endif

#if defined(ET_RTOL)
const float et_rtol = ET_RTOL;
#else
const float et_rtol = 0.01;
#endif

#endif

#if defined(ET_NUM_INFERENCES)
const int num_inferences = ET_NUM_INFERENCES;
#else
const int num_inferences = 1;
#endif

/**
 * The temp_allocation_pool is used for allocating temporary data during kernel
 * or delegate execution. This will be reset after each kernel or delegate call.
 * Currently a MemoryAllocator is used but a PlatformMemoryAllocator is probably
 * a better fit.
 *
 * The Corstone-300/Corstone-320 platforms have 2MB/4MB of SRAM respectively.
 * For Shared_Sram, ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE is
 * 2MB and the linker script places the .bss.tensor_arena symbol in the SRAM.
 * For Dedicated_Sram, the .bss.tensor_arena symbol is placed in the DDR in the
 * linker script. Hence, we allocate 128MB in DDR and 384KB in the SRAM
 * (.bss.ethosu_scratch is placed in the SRAM). The examples/arm/CMakeLists.txt
 * contains the logic for the sizes of
 * ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE and
 * ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
 */
constexpr size_t temp_allocation_pool_size = 0;
//    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char* temp_allocation_pool = nullptr;
// unsigned char __attribute__((
//     section(".bss.tensor_arena"),
//     aligned(16))) temp_allocation_pool[temp_allocation_pool_size];
#if defined(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
extern "C" {
size_t ethosu_fast_scratch_size =
    ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((section(".bss.ethosu_scratch"), aligned(16)))
dedicated_sram[ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
unsigned char* ethosu_fast_scratch = dedicated_sram;
}
#endif

void et_pal_init(void) {
  // Enable ARM PMU Clock
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk; // Trace enable
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

/**
 * Implementation of the et_pal_<funcs>()
 *
 * This functions are hardware adaption type of functions for things like
 * time/logging/memory allocation that could call your RTOS or need to to
 * be implemnted in some way.
 */

ET_NORETURN void et_pal_abort(void) {
#if !defined(SEMIHOSTING)
  __builtin_trap();
#else
  _exit(-1);
#endif
}

et_timestamp_t et_pal_current_ticks(void) {
  return ARM_PMU_Get_CCNTR();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't know the CPU freq for your target and justs cycles in the
  // FVP for et_pal_current_ticks() we return a conversion ratio of 1
  return {1, 1};
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(
      stderr,
      "%c [executorch:%s:%zu %s()] %s\n",
      level,
      filename,
      line,
      function,
      message);
}

/**
 * Dynamic memory allocators intended to be used by temp_allocator
 * to implement malloc()/free() type of allocations.
 * Currenyly not used.
 */

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}

namespace {

/// Lightweight heapless container that constructs and stores a T in-place.
/// Useful when you want to avoid heap allocations but need to delay
/// construction.
template <typename T>
class Box {
 public:
  Box() = default;

  ~Box() {
    if (has_value) {
      ptr()->~T();
    }
  }

  Box(const Box&) = delete;
  Box& operator=(const Box&) = delete;

  /// Destructs the already contained object if it's present and initialize a
  /// new contained object while forwarding its constructor arguments.
  template <typename... Args>
  void reset(Args&&... args) {
    if (has_value) {
      // Destroy the already contained object.
      reinterpret_cast<T*>(mem)->~T();
    }
    // Init the new object.
    new (mem) T(std::forward<Args>(args)...);
    has_value = true;
  }

  /// Returns a reference to the contained object.
  T& value() {
    return *ptr();
  }

  /// Returns a const reference to the contained object.
  const T& value() const {
    return *ptr();
  }

  T* operator->() {
    return ptr();
  }

  const T* operator->() const {
    return ptr();
  }

 private:
  alignas(T) uint8_t mem[sizeof(T)];
  bool has_value = false;

  T* ptr() {
    return reinterpret_cast<T*>(mem);
  }

  const T* ptr() const {
    return reinterpret_cast<const T*>(mem);
  }
};

template <typename ValueType>
void fill_tensor_with_default_value(Tensor& tensor) {
  ValueType fill_value{};
  if constexpr (std::is_same_v<ValueType, bool>) {
    fill_value = true;
  } else {
    fill_value = ValueType(1);
  }

  ValueType* data_ptr = tensor.mutable_data_ptr<ValueType>();
  std::fill(data_ptr, data_ptr + tensor.numel(), fill_value);
}

Error prepare_input_tensors(
    Method& method,
    MemoryAllocator& allocator,
    const std::vector<std::pair<char*, size_t>>& input_buffers) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;

#if defined(SEMIHOSTING)
  ET_CHECK_OR_RETURN_ERROR(
      input_buffers.size() > 0 && num_inputs == input_buffers.size(),
      InvalidArgument,
      "Wrong number of inputs allocated compared to method");
#endif

  EValue* input_evalues = allocator.allocateList<EValue>(num_inputs);
  ET_CHECK_OR_RETURN_ERROR(
      input_evalues != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for input evalues.");

  Error err = method.get_inputs(input_evalues, num_inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(err);

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    err = Error::Ok;
    if (input_buffers.size() > 0) {
      auto [buffer, buffer_size] = input_buffers.at(i);
      if (buffer_size != tensor_meta->nbytes()) {
        ET_LOG(
            Error,
            "input size (%d) and tensor size (%d) mismatch!",
            buffer_size,
            tensor_meta->nbytes());
        err = Error::InvalidArgument;
      } else if (input_evalues[i].isTensor()) {
        // Copy the data from the input buffer to the tensor
        Tensor& tensor = input_evalues[i].toTensor();
        std::memcpy(tensor.mutable_data_ptr<int8_t>(), buffer, buffer_size);
      }
    }

    // If input_buffers.size <= 0, we don't have any input, fill it with 1's.
    if (input_buffers.size() <= 0) {
      if (input_evalues[i].isTensor()) {
        Tensor& tensor = input_evalues[i].toTensor();
        switch (tensor.scalar_type()) {
#define HANDLE_SCALAR_TYPE(cpp_type, scalar_name)     \
  case ScalarType::scalar_name:                       \
    fill_tensor_with_default_value<cpp_type>(tensor); \
    break;
          ET_FORALL_SCALAR_TYPES(HANDLE_SCALAR_TYPE)
#undef HANDLE_SCALAR_TYPE
          default:
            ET_LOG(
                Error,
                "Unhandled ScalarType %s",
                toString(tensor.scalar_type()));
            err = Error::InvalidArgument;
            break;
        }
      } else {
        printf("Input[%d]: Not Tensor\n", i);
      }
    }
  }

  return err;
}

#if defined(SEMIHOSTING)

std::pair<char*, size_t> read_binary_file(
    const char* filename,
    MemoryAllocator& allocator) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    ET_LOG(
        Fatal,
        "Could not open file %s (errno: %d) for reading, exiting!",
        filename,
        errno);
    return std::make_pair(nullptr, 0);
  }

  fseek(fp, 0, SEEK_END);
  auto file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char* buffer = static_cast<char*>(allocator.allocate(file_size));
  if (buffer == nullptr) {
    ET_LOG(
        Fatal, "Failed to allocate input file size:%zu", (uint32_t)file_size);
    return std::make_pair(nullptr, 0);
  }
  auto read_size = fread(buffer, 1, file_size, fp);
  if (read_size != file_size) {
    ET_LOG(
        Info,
        "Failed to read whole file (%), read %zu bytes!",
        filename,
        read_size);
  }
  fclose(fp);
  return std::make_pair(buffer, read_size);
}
#endif

/// Holds all state needed for setup and run phases
struct RunnerContext {
  RunnerContext() = default;
  RunnerContext(const RunnerContext& ctx) = delete;
  RunnerContext& operator=(const RunnerContext& ctx) = delete;

  const char* method_name = nullptr;
  size_t planned_buffer_memsize = 0;
  size_t method_loaded_memsize = 0;
  size_t executor_membase = 0;
  size_t program_data_len = 0;
  size_t input_memsize = 0;
  size_t pte_size = 0;
  bool bundle_io = false;
  Box<BufferDataLoader> loader;
  Box<Program> program;
  Box<ArmMemoryAllocator> method_allocator;
  Box<ArmMemoryAllocator> temp_allocator;
  std::vector<Span<uint8_t>> planned_spans;
  Box<HierarchicalAllocator> planned_memory;
  Box<MemoryManager> memory_manager;
  Box<Result<Method>> method;
#if defined(ET_EVENT_TRACER_ENABLED)
  Box<ETDumpGen> etdump_gen;
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
  void* debug_buffer;
#endif
#endif
#if defined(SEMIHOSTING)
  Box<ArmMemoryAllocator> input_file_allocator;
  const char* output_basename = nullptr;
#endif
};

void runner_init(
    RunnerContext& ctx,
    std::vector<std::pair<char*, size_t>> input_buffers,
    size_t pte_size) {
  // Find the offset to the embedded Program.
  const void* program_data = model_pte;
  ctx.program_data_len = pte_size;
  ctx.pte_size = pte_size;

#if defined(ET_BUNDLE_IO)
  ctx.bundle_io = executorch::bundled_program::is_bundled_program(
      reinterpret_cast<void*>(model_pte), ctx.pte_size);
  if (ctx.bundle_io) {
    // BundleIO bpte is provided, dig out the actual model from the data area
    Error status = executorch::bundled_program::get_program_data(
        reinterpret_cast<void*>(model_pte),
        ctx.pte_size,
        &program_data,
        &ctx.program_data_len);

    ET_CHECK_MSG(
        status == Error::Ok,
        "get_program_data() from bundle PTE failed: 0x%x",
        (unsigned int)status);
  }
#endif
  ctx.loader.reset(program_data, ctx.program_data_len);
  auto& loader = ctx.loader.value();
  ET_LOG(Info, "PTE Model data loaded. Size: %zu bytes.", ctx.program_data_len);

  // Parse the program file. This is immutable, and can also be reused
  // between multiple execution invocations across multiple threads.
  Result<Program> program_result = Program::load(&loader);
  ET_CHECK_MSG(
      program_result.ok(),
      "Program loading failed @ %p: 0x%" PRIx32,
      program_data,
      program_result.error());
  ctx.program.reset(std::move(program_result.get()));
  Program& program = ctx.program.value();

  ET_LOG(Info, "Model buffer loaded, has %zu methods", program.num_methods());

  {
    const auto method_name_result = program.get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    ctx.method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", ctx.method_name);

  Result<MethodMeta> method_meta = program.method_meta(ctx.method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Info,
        "Failed to get method_meta for %s: 0x%x",
        ctx.method_name,
        (unsigned int)method_meta.error());
  }

  ET_LOG(
      Info,
      "Setup Method allocator pool. Size: %zu bytes.",
      method_allocation_pool_size);

  ctx.method_allocator.reset(
      method_allocation_pool_size, method_allocation_pool);

  ctx.planned_spans.clear();
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  ctx.planned_spans.reserve(num_memory_planned_buffers);
  size_t planned_buffer_membase = ctx.method_allocator->used_size();
  ET_LOG(
      Info,
      "Method meta, has %zu instructions",
      method_meta->num_instructions());
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);

    /* Move to it's own allocator when MemoryPlanner is in place. */
    /* Ethos-U driver requires 16 bit alignment. */
    uint8_t* buffer = reinterpret_cast<uint8_t*>(
        ctx.method_allocator->allocate(buffer_size, 16UL));
    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate memory for memory planned buffer size %zu",
        buffer_size);
    ctx.planned_spans.push_back({buffer, buffer_size});
  }

  ctx.planned_buffer_memsize =
      ctx.method_allocator->used_size() - planned_buffer_membase;

  Span<Span<uint8_t>> planned_memory_span;
  if (!ctx.planned_spans.empty()) {
    planned_memory_span =
        Span<Span<uint8_t>>(ctx.planned_spans.data(), ctx.planned_spans.size());
  }
  ctx.planned_memory.reset(planned_memory_span);

  ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);

  ctx.memory_manager.reset(
      &ctx.method_allocator.value(),
      &ctx.planned_memory.value(),
      &ctx.temp_allocator.value());

  size_t method_loaded_membase = ctx.method_allocator->used_size();

  executorch::runtime::EventTracer* event_tracer_ptr = nullptr;

#if defined(ET_EVENT_TRACER_ENABLED)
  ET_LOG(Info, "Setting up ETDump");
  ctx.etdump_gen.reset();
  event_tracer_ptr = &ctx.etdump_gen.value();

#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
  // Alloc debug buffer and create if and only if we need to log intermediate
  // tensor outputs
  ctx.debug_buffer = ctx.method_allocator->allocate(ET_DEBUG_BUFFER_SIZE, 16);
  if (ctx.debug_buffer != nullptr) {
    Span<uint8_t> debug_buffer_span(
        (uint8_t*)ctx.debug_buffer, ET_DEBUG_BUFFER_SIZE);

    Result<bool> result =
        ctx.etdump_gen.value().set_debug_buffer(debug_buffer_span);

    if (result.ok()) {
      // Everything worked, we got the buffer setup, lets enable output logging
      // depending on the compile flag ET_DUMP_INTERMEDIATE_OUTPUTS e.g.
      // kIntermediateOutputs or kProgramOutputs
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS)
      ET_LOG(
          Info,
          "ETDump: Allocated intermediate output buffer size: %d at 0x%p",
          ET_DEBUG_BUFFER_SIZE,
          ctx.debug_buffer);
      ctx.etdump_gen.value().set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
#else // defined(ET_DUMP_INTERMEDIATE_OUTPUTS)
      ET_LOG(
          Info,
          "ETDump: Allocated output buffer size: %d at 0x%p",
          ET_DEBUG_BUFFER_SIZE,
          ctx.debug_buffer);
      ctx.etdump_gen.value().set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kProgramOutputs);
#endif // defined(ET_DUMP_INTERMEDIATE_OUTPUTS)

    } else {
      // set_debug_buffer() failed
      // Here we would free ctx.debug_buffer if it was possible, but we can't as
      // the allocator don't support it.
      ctx.debug_buffer = nullptr;
      ET_LOG(
          Error,
          "ETDump: Could not set_debug_buffer() for output buffer size %zu error:0x%" PRIx32,
          ET_DEBUG_BUFFER_SIZE,
          result.error());
    }
  } else {
    // debug buffer allocation failed
    ET_LOG(
        Error,
        "ETDump: Could not allocate memory for output buffer size %zu",
        ET_DEBUG_BUFFER_SIZE);
  }
#endif // defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
#endif // defined(ET_EVENT_TRACER_ENABLED)

  ctx.method.reset(program.load_method(
      ctx.method_name, &ctx.memory_manager.value(), event_tracer_ptr));

  if (!ctx.method->ok()) {
    ET_LOG(
        Info,
        "Loading of method %s failed with status 0x%" PRIx32,
        ctx.method_name,
        ctx.method->error());
  }
  ctx.method_loaded_memsize =
      ctx.method_allocator->used_size() - method_loaded_membase;
  ET_LOG(Info, "Method '%s' loaded.", ctx.method_name);

  ET_LOG(Info, "Preparing inputs...");
  size_t input_membase = ctx.method_allocator->used_size();

#if defined(ET_BUNDLE_IO)
  if (ctx.bundle_io) {
    // Get inputs from bundled IO ".bpte" data
    // Useful for testing
    ET_LOG(Info, "Input testset[%d] from bundled bpte", testset_idx);
    Error status = executorch::bundled_program::load_bundled_input(
        *ctx.method.value(), model_pte, testset_idx);
    ET_CHECK_MSG(
        status == Error::Ok,
        "load_bundled_input failed with status 0x%" PRIx32,
        status);
  } else
#endif
  {
    Error status = ::prepare_input_tensors(
        *ctx.method.value(), ctx.method_allocator.value(), input_buffers);
    ET_CHECK_MSG(
        status == Error::Ok, "Failed to prepare inputs 0x%" PRIx32, status);
  }
#if defined(ET_LOG_DUMP_INPUT)
  {
    std::vector<EValue> inputs((*ctx.method.value())->inputs_size());
    ET_LOG(Info, "%zu inputs: ", inputs.size());
    Error status = ctx.method.value()->get_inputs(inputs.data(), inputs.size());
    ET_CHECK(status == Error::Ok);

    for (int i = 0; i < inputs.size(); ++i) {
      if (inputs[i].isTensor()) {
        Tensor tensor = inputs[i].toTensor();
        // The output might be collected and parsed so printf() is used instead
        // of ET_LOG() here
        for (int j = 0; j < tensor.numel(); ++j) {
          if (tensor.scalar_type() == ScalarType::Int) {
            printf(
                "Input[%d][%d]: (int) %d\n",
                i,
                j,
                tensor.const_data_ptr<int>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Float) {
            printf(
                "Input[%d][%d]: (float) %f\n",
                i,
                j,
                tensor.const_data_ptr<float>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Char) {
            printf(
                "Input[%d][%d]: (char) %d\n",
                i,
                j,
                tensor.const_data_ptr<int8_t>()[j]);
          } else if (tensor.scalar_type() == ScalarType::Bool) {
            printf(
                "Input[%d][%d]: (bool) %s (0x%x)\n",
                i,
                j,
                tensor.const_data_ptr<int8_t>()[j] ? "true" : "false",
                tensor.const_data_ptr<int8_t>()[j]);
          }
        }
      } else {
        printf("Input[%d]: Not Tensor\n", i);
      }
    }
  }
#endif
  ctx.input_memsize = ctx.method_allocator->used_size() - input_membase;
  ctx.executor_membase = ctx.method_allocator->used_size();

  ET_LOG(Info, "Input prepared.");
}

void log_mem_status(RunnerContext& ctx) {
  size_t executor_memsize =
      ctx.method_allocator->used_size() - ctx.executor_membase;

#if defined(ET_MODEL_PTE_ADDR)
  ET_LOG(
      Info,
      "model_pte_program_size:     %zu bytes. (pte size unknown when not baked into elf)",
      ctx.program_data_len);
  ET_LOG(
      Info,
      "model_pte_loaded_size:      %zu bytes. (pte size unknown when not baked into elf)",
      ctx.pte_size);
#else
  ET_LOG(Info, "model_pte_program_size:     %zu bytes.", ctx.program_data_len);
  ET_LOG(Info, "model_pte_loaded_size:      %zu bytes.", ctx.pte_size);
#endif

#if defined(SEMIHOSTING)
  if (ctx.input_file_allocator->size() > 0) {
    ET_LOG(
        Info,
        "input_file_allocator_used: %zu / %zu free: %zu ( used: %zu %% ) ",
        ctx.input_file_allocator->used_size(),
        ctx.input_file_allocator->size(),
        ctx.input_file_allocator->free_size(),
        100 * ctx.input_file_allocator->used_size() /
            ctx.input_file_allocator->size());
  }
#endif
  if (ctx.method_allocator->size() != 0) {
    size_t method_allocator_used = ctx.method_allocator->used_size();
    ET_LOG(
        Info,
        "method_allocator_used:     %zu / %zu  free: %zu ( used: %zu %% ) ",
        method_allocator_used,
        ctx.method_allocator->size(),
        ctx.method_allocator->free_size(),
        100 * method_allocator_used / ctx.method_allocator->size());
    ET_LOG(
        Info,
        "method_allocator_planned:  %zu bytes",
        ctx.planned_buffer_memsize);
    ET_LOG(
        Info,
        "method_allocator_loaded:   %zu bytes",
        ctx.method_loaded_memsize);
    ET_LOG(Info, "method_allocator_input:    %zu bytes", ctx.input_memsize);
    ET_LOG(Info, "method_allocator_executor: %zu bytes", executor_memsize);
  }
  if (ctx.temp_allocator->size() > 0) {
    ET_LOG(Info, "temp_allocator:            %zu", ctx.temp_allocator->size());
  }
#if defined(ET_EVENT_TRACER_ENABLED)
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
  if (ctx.debug_buffer != nullptr) {
    size_t outputdump_len = ctx.etdump_gen->get_data_sink()->get_used_bytes();
    ET_LOG(
        Info,
        "ETDump_outputs_buffer:     %zu / %zu free: %zu ( used: %zu %% ) ",
        outputdump_len,
        ET_DEBUG_BUFFER_SIZE,
        ET_DEBUG_BUFFER_SIZE - outputdump_len,
        100 * outputdump_len / ET_DEBUG_BUFFER_SIZE);
  }
#endif
#endif
}

void print_outputs(RunnerContext& ctx) {
  std::vector<EValue> outputs(ctx.method.value()->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  Error status =
      ctx.method.value()->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  // Print the outputs.
  for (int i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      Tensor tensor = outputs[i].toTensor();
#if !defined(SEMIHOSTING)
#if defined(ET_LOG_DUMP_OUTPUT)
      // The output might be collected and parsed so printf() is used instead
      // of ET_LOG() here
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
#endif
#else //! defined(SEMIHOSTING)
      char out_filename[255];
      snprintf(out_filename, 255, "%s-%d.bin", ctx.output_basename, i);
      ET_LOG(Info, "Writing output to file: %s", out_filename);
      FILE* out_file = fopen(out_filename, "wb");
      auto written_size =
          fwrite(tensor.const_data_ptr<char>(), 1, tensor.nbytes(), out_file);
      fclose(out_file);
#endif //! defined(SEMIHOSTING)
    } else {
      printf("Output[%d]: Not Tensor\n", i);
    }
  }
}

void write_etdump(RunnerContext& ctx) {
#if defined(ET_EVENT_TRACER_ENABLED)
#if !defined(SEMIHOSTING)
  // Dump the etdump data containing profiling/debugging data to the serial line
  // base64 encoded
  ETDumpResult result = ctx.etdump_gen->get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    // On a device with no file system we can't just write it out
    // to the file-system so we base64 encode it and dump it on the log.
    bool dump_outputs = false;
    int mode = base64_enc_modifier_padding | base64_dec_modifier_skipspace;
    size_t etdump_len = result.size;
    size_t encoded_etdump_len = base64_encoded_size(etdump_len, mode);
    size_t base64buffer_len = encoded_etdump_len;
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
    // Make base64 buffer fit both so it can be reused istead of allocating two
    // buffers.
    size_t outputdump_len = 0;
    size_t encoded_outputdump_len = 0;
    if (ctx.debug_buffer != nullptr) {
      outputdump_len = ctx.etdump_gen->get_data_sink()->get_used_bytes();
      if (outputdump_len > 0) {
        encoded_outputdump_len = base64_encoded_size(outputdump_len, mode);
        if (encoded_outputdump_len > 0) {
          base64buffer_len =
              std::max(encoded_etdump_len, encoded_outputdump_len);
          dump_outputs = true;
        } else {
          ET_LOG(
              Error,
              "Problem getting the size of the base64 ETDump output buffers");
        }
      } else {
        ET_LOG(Error, "No ETDump output buffers saved in the data area");
      }
    }
#endif
    ET_LOG(Info, "[base64] buffer size: %d", base64buffer_len);

    uint8_t* encoded_buf = reinterpret_cast<uint8_t*>(
        ctx.method_allocator->allocate(base64buffer_len + 1));
    if (encoded_buf != nullptr) {
      int ret;
      const char* debug_buffer_flag = "";
      printf("#[RUN THIS]\n");
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
      if (dump_outputs) {
        ret = base64_encode(
            encoded_buf,
            (uint8_t*)ctx.debug_buffer,
            &encoded_outputdump_len,
            &outputdump_len,
            mode);
        encoded_buf[encoded_outputdump_len] = 0x00; // Ensure null termination
        printf("# Writing debug_buffer.bin [base64]\n");
        printf("echo \"%s\" | base64 -d >debug_buffer.bin\n", encoded_buf);
        debug_buffer_flag = "--debug_buffer_path debug_buffer.bin";
      }
#endif
      ret = base64_encode(
          encoded_buf,
          (uint8_t*)result.buf,
          &encoded_etdump_len,
          &etdump_len,
          mode);
      encoded_buf[encoded_etdump_len] = 0x00; // Ensure null termination
      printf("# Writing etdump.bin [base64]\n");
      printf("echo \"%s\" | base64 -d >etdump.bin\n", encoded_buf);

      printf("# Generate cpu cycle table with:\n");
      printf(
          "python3 -m devtools.inspector.inspector_cli --etdump_path etdump.bin %s --source_time_scale cycles --target_time_scale cycles\n",
          debug_buffer_flag);
      printf("#[END]\n");

    } else {
      ET_LOG(
          Error,
          "Could not allocate memory etdump base64 encoding size %zu",
          encoded_etdump_len + 1);
    }
  }
#else // !defined(SEMIHOSTING)
#if defined(ET_DUMP_INTERMEDIATE_OUTPUTS) || defined(ET_DUMP_OUTPUTS)
  if (ctx.debug_buffer != nullptr) {
    // Dump the etdump outputs data to a file.
    size_t outputdump_len = ctx.etdump_gen->get_data_sink()->get_used_bytes();
    const char* etdump_output_filename = "debug_buffer.bin";
    ET_LOG(
        Info,
        "Writing etdump debug_buffer to file: %s",
        etdump_output_filename);
    FILE* f = fopen(etdump_output_filename, "w+");
    fwrite((uint8_t*)ctx.debug_buffer, 1, outputdump_len, f);
    fclose(f);
  }
#endif

  // Dump the etdump data containing profiling/debugging data to a file.
  etdump_result result = ctx.etdump_gen->get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    // On a device with a file system we can just write it out
    // to the file-system.
    const char* etdump_filename = "etdump.bin";
    ET_LOG(Info, "Writing etdump to file: %s", etdump_filename);
    FILE* f = fopen(etdump_filename, "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  }
#endif // !defined(SEMIHOSTING)
#endif // defined(ET_EVENT_TRACER_ENABLED)
}

bool verify_result(RunnerContext& ctx, const void* model_pte) {
  bool model_ok = false;
#if defined(ET_BUNDLE_IO)
  if (ctx.bundle_io) {
    // Check result
    ErrorStats stats = compute_method_output_error_stats(
        *ctx.method.value(), model_pte, testset_idx);
    if (stats.status == Error::Ok) {
      ET_LOG(Info, "=== Error stats for testset %d ===", testset_idx);
      ET_LOG(Info, " mean_absolute_error: %f", stats.mean_abs_error);
      ET_LOG(Info, " max_absolute_error:  %f", stats.max_abs_error);
      ET_LOG(Info, " mean_relative_error: %f", stats.mean_relative_error);
      ET_LOG(Info, " max_relative_error:  %f", stats.max_relative_error);
    } else {
      ET_LOG(
          Info,
          "=== Error calculating stats for testset %d ERROR:%d ===",
          testset_idx,
          stats.status);
    }

    // Verify the result.
    Error status = verify_method_outputs(
        *ctx.method.value(), model_pte, testset_idx, et_rtol, et_atol);
    if (status == Error::Ok) {
      ET_LOG(Info, "Model output match expected BundleIO bpte ref data.");
      ET_LOG(Info, "TEST: BundleIO index[%d] Test_result: PASS", testset_idx);
      model_ok = true;
    } else {
      ET_LOG(
          Error,
          "Model output don't match expected BundleIO bpte ref data. rtol=%f atol=%f",
          et_rtol,
          et_atol);
      ET_LOG(Error, "TEST: BundleIO index[%d] Test_result: FAIL", testset_idx);
      ET_LOG(
          Error, "Bundle verification failed with status 0x%" PRIx32, status);
      model_ok = false;
    }
  } else {
    // No checking done, assume true
    model_ok = true;
  }
#else // defined(ET_BUNDLE_IO)
  (void)ctx;
  (void)model_pte;
  // No checking done, assume true
  model_ok = true;
#endif // defined(ET_BUNDLE_IO)
  return model_ok;
}

bool run_model(RunnerContext& ctx, const void* model_pte) {
  Error status;
  ET_LOG(Info, "Starting running %d inferences...", num_inferences);
  int n = 0;
  StartMeasurements();
  for (n = 0; n < num_inferences; n++) {
    ET_LOG(Debug, "Running inference number %d", n);
    // Run the model.
    status = ctx.method.value()->execute();
    if (status != Error::Ok) {
      break;
    }
    // Reset the temporary allocator holding the scratch buffer between
    // inferences. We want to reuse the temp_allocator between inferences of the
    // same Ethos-U custom delegate, not allocate memory with every new
    // inference.
    ctx.temp_allocator.reset(temp_allocation_pool_size, temp_allocation_pool);
  }
  StopMeasurements(n);

  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      ctx.method_name,
      status);

  ET_LOG(Info, "%d inferences finished", num_inferences);
  print_outputs(ctx);
  bool model_ok = verify_result(ctx, model_pte);
  ET_LOG(Info, "Model run: %d", model_ok);

  return model_ok;
}

} // namespace

int main(int argc, const char* argv[]) {
#if defined(SEMIHOSTING)
  ET_LOG(Info, "Running executor with parameter:");
  if (argc < 7) {
    ET_LOG(Fatal, "Not right number of parameters!");
    ET_LOG(
        Fatal,
        "app -m model.pte -i input.bin [-i input2.bin] -o output_basename");
    ET_LOG(Fatal, "Exiting!");
    _exit(1);
  }
  ET_LOG(Info, "   %s", argv[0]);
  for (int i = 1; i < argc; i++) {
    ET_LOG(Info, "   %s %s", argv[i], argv[++i]);
  }
#else
  (void)argc;
  (void)argv;
#endif

  executorch::runtime::runtime_init();
  std::vector<std::pair<char*, size_t>> input_buffers;

#if defined(ET_MODEL_PTE_ADDR)
  // pte not in a known array but just on a memory/flash address
  // As we dont know the size we pick something big enough
  // Actual model is read from this area.
  size_t pte_size = 0x10000000;
#else
  size_t pte_size = sizeof(model_pte);
#endif

  RunnerContext ctx;

#if defined(SEMIHOSTING)
  ctx.input_file_allocator.reset(
      input_file_allocation_pool_size, input_file_allocation_pool);

  /* parse input parameters */
  for (int i = 0; i < argc; i++) {
    size_t nbr_inputs = 0;
    if (std::strcmp(argv[i], "-i") == 0) {
      // input file, read the data into memory
      const char* input_tensor_filename = argv[++i];
      ET_LOG(
          Info,
          "Reading input tensor %d from file %s",
          ++nbr_inputs,
          input_tensor_filename);
      auto [buffer, buffer_size] = read_binary_file(
          input_tensor_filename, ctx.input_file_allocator.value());
      if (buffer == nullptr) {
        ET_LOG(
            Error,
            "Reading input tensor %d from file %s ERROR Out of memory",
            nbr_inputs,
            input_tensor_filename);
        _exit(1);
      }
      input_buffers.push_back(std::make_pair(buffer, buffer_size));
    } else if (std::strcmp(argv[i], "-m") == 0) {
      const char* pte_filename = argv[++i];
      ET_LOG(Info, "Reading pte model from file %s", pte_filename);
      auto [buffer, buffer_size] =
          read_binary_file(pte_filename, ctx.input_file_allocator.value());
      if (buffer == nullptr) {
        ET_LOG(
            Error,
            "Reading pte model from file %s ERROR Out of memory",
            pte_filename);
        _exit(1);
      }

      // Store the model data with the same variable as if it was loaded
      // from compiled in location.
      model_pte = buffer;
      pte_size = buffer_size;
    } else if (std::strcmp(argv[i], "-o") == 0) {
      // store the base filename to write output to.
      ctx.output_basename = argv[++i];
    }
  }
#endif

  // Byte 4-7 is usually a nice magic number that could be good to print to make
  // sure it's OK ETxx for PTE and BPxx for bundled pte where xx is a number.
  ET_LOG(
      Info,
      "PTE @ %p [----%c%c%c%c]",
      model_pte,
      model_pte[4],
      model_pte[5],
      model_pte[6],
      model_pte[7]);

  runner_init(ctx, input_buffers, pte_size);
  bool model_ok = run_model(ctx, model_pte);
  ET_LOG(Info, "Model run: %d", model_ok);

  log_mem_status(ctx);
  write_etdump(ctx);

  ET_CHECK_MSG(model_ok == true, "Problem running model");

  ET_LOG(Info, "Program complete, exiting.");
#if defined(SEMIHOSTING)
  _exit(0);
#endif
  ET_LOG(Info, "\04");
  return 0;
}
