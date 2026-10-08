/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2023-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Common Arm backend for Ethos-U. Please see
 * EthosUBackend_Cortex_*.cpp for specific backends.
 */

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

// Overridable memcpy for copying inputs and outputs to and from scratch.
// Default (weak) implementation in EthosUBackend_IoMemcpy.cpp does
// std::memcpy. Firmware targets can supply a strong override (e.g. routing
// through a DMA engine) to reduce CPU memcpy load on the host MCU.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size);

using namespace std;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace executorch {
namespace backends {
namespace arm {

namespace {

using printf_size_t = unsigned long;

Error validate_et_and_vela_tensors(
    const executorch::aten::Tensor& tensor,
    const VelaIO& io) {
  if (io.elem_size != tensor.element_size()) {
    ET_LOG(
        Error,
        "Element size mismatch: tensor %lu, Vela %d",
        static_cast<printf_size_t>(tensor.element_size()),
        io.elem_size);
    return Error::InvalidProgram;
  }

  // init() validates the dimensions and rules out byte-count overflow.
  const size_t io_bytes = std::accumulate(
      std::begin(io.shape),
      std::end(io.shape),
      static_cast<size_t>(io.elem_size),
      std::multiplies<size_t>{});
  if (io_bytes != tensor.nbytes()) {
    ET_LOG(
        Error,
        "Byte size mismatch: tensor %lu, Vela %lu",
        static_cast<printf_size_t>(tensor.nbytes()),
        static_cast<printf_size_t>(io_bytes));
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

} // namespace

extern "C" {
void __attribute__((weak)) EthosUBackend_execute_begin() {}
void __attribute__((weak)) EthosUBackend_execute_end() {}
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
void __attribute__((weak)) EthosUBackend_delegate_begin(const void*) {}
void __attribute__((weak)) EthosUBackend_delegate_end() {}
#endif
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
void __attribute__((weak)) EthosUBackend_input_memcpy(size_t) {}
void __attribute__((weak)) EthosUBackend_output_memcpy(size_t) {}
#endif
__attribute__((weak)) unsigned char* ethosu_fast_scratch = nullptr;
__attribute__((weak)) size_t ethosu_fast_scratch_size = 0;
}

class EthosUBackendExecuteCallbacks {
 public:
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
  explicit EthosUBackendExecuteCallbacks(const void* handle) {
    EthosUBackend_execute_begin();
    EthosUBackend_delegate_begin(handle);
  }
#else
  EthosUBackendExecuteCallbacks() {
    EthosUBackend_execute_begin();
  }
#endif
  ~EthosUBackendExecuteCallbacks() {
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    EthosUBackend_delegate_end();
#endif
    EthosUBackend_execute_end();
  }
};

class EthosUBackend final : public ::executorch::runtime::BackendInterface {
 public:
  EthosUBackend() {}

  ~EthosUBackend() = default;

  virtual bool is_available() const override {
    // TODO: revise to use a register check/init function
    return 1;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
#if defined(ET_EVENT_TRACER_ENABLED)
    EventTracer* event_tracer = context.event_tracer();
    EventTracerEntry event_tracer_local_scope;
#endif

    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::init()processed_data");
    const char* data = static_cast<const char*>(processed->data());
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);

    ET_LOG(Info, "data:%p", data);
    size_t size = processed->size();

    // Verify format of vela_bin
    if (vela_bin_validate(data, size) == false) {
      ET_LOG(Error, "Malformed vela_bin_stream found");
      return Error::InvalidProgram;
    }

    MemoryAllocator* allocator = context.get_runtime_allocator();
    ExecutionHandle* handle = allocator->allocateInstance<ExecutionHandle>();
    if (handle == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    new (handle) ExecutionHandle();

    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::init()vela_bin_read()");
    const Error read_status = vela_bin_read(
        data, size, context.get_named_data_map(), &handle->handles);
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);
    if (read_status != Error::Ok) {
      handle->~ExecutionHandle();
      return read_status;
    }

    const VelaIOs* outputs = handle->handles.outputs;
    const int output_count = outputs ? outputs->count : 0;
    for (int i = 0; i < output_count; ++i) {
      const VelaIO& output_io = outputs->io[i];
      if (output_io.elem_size <= 0) {
        ET_LOG(Error, "Ethos-U output %d has an invalid element size", i);
        handle->~ExecutionHandle();
        return Error::InvalidProgram;
      }
      size_t io_bytes = static_cast<size_t>(output_io.elem_size);
      for (int dim : output_io.shape) {
        if (dim < 0 ||
            (dim > 0 &&
             io_bytes > std::numeric_limits<size_t>::max() /
                     static_cast<size_t>(dim))) {
          ET_LOG(Error, "Ethos-U output %d has an invalid shape", i);
          handle->~ExecutionHandle();
          return Error::InvalidProgram;
        }
        io_bytes *= static_cast<size_t>(dim);
      }
    }

    const Error platform_status =
        platform_init(compile_specs, allocator, handle);
    if (platform_status != Error::Ok) {
      handle->~ExecutionHandle();
      return platform_status;
    }

    // Return the same buffer we were passed - this data will be
    // executed directly
    return handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      Span<EValue*> args) const override {
#if defined(ET_EVENT_TRACER_ENABLED)
    EventTracer* event_tracer = context.event_tracer();
    EventTracerEntry event_tracer_local_scope;
#endif

    EXECUTORCH_PROF_SCOPE(event_tracer, "EthosUBackend::execute()");

    // CollectArm_CPU_Cycles is just used to save the numbers of CPU cycles
    // used, If etdump is used the EXECUTORCH_PROF_SCOPE() above will do the
    // same. If not, this is a cheap way of getting some stats and the
    // CollectArm_CPU_Cycles object can safely be removed in production code.
    //
    // The EthosUBackendExecuteCallbacks class uses the C++
    // constructor/destructor to make sure that EthosUBackend_execute_begin()
    // and EthosUBackend_execute_end() is called while CollectArm_CPU_Cycles is
    // in scope. e.g. We meassure from now until we exit this metod (in any way
    // we might do it).
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
    EthosUBackendExecuteCallbacks CollectArm_CPU_Cycles(input_handle);
#else
    EthosUBackendExecuteCallbacks CollectArm_CPU_Cycles;
#endif

    ExecutionHandle* execution_handle =
        static_cast<ExecutionHandle*>(input_handle);
    VelaHandles handles = execution_handle->handles;

    const int input_count = handles.inputs ? handles.inputs->count : 0;
    const int output_count = handles.outputs ? handles.outputs->count : 0;

    // Output buffers must match Vela's element sizes and byte counts.
    for (int i = 0; i < output_count; ++i) {
      const Error status = validate_et_and_vela_tensors(
          args[input_count + i]->toTensor(), handles.outputs->io[i]);
      if (status != Error::Ok) {
        ET_LOG(
            Error, "Ethos-U output %d does not match its Vela descriptor", i);
        return status;
      }
    }

    char* ethosu_scratch = nullptr;
    if (needs_scratch_allocation()) {
      MemoryAllocator* temp_allocator = context.get_temp_allocator();
      // Use a temporary allocator for the intermediate tensors of the
      // computation. The allocator is released in runtime/executor/method.cpp
      // at the end of the execution of the Ethos-U custom delegate. Ethos-U
      // driver requires 16 bit alignment.
      ethosu_scratch = static_cast<char*>(
          temp_allocator->allocate(handles.scratch_data_size, 16UL));
      if (ethosu_scratch == nullptr) {
        ET_LOG(
            Error,
            "Failed to allocate scratch buffer of %lu bytes from temp_allocator",
            static_cast<printf_size_t>(handles.scratch_data_size));
        return Error::MemoryAllocationFailed;
      }
    }

    ET_LOG(
        Debug,
        "Running program data:\n  cmd %p %lu\n  weight %p %lu\n  scratch %p %lu\n  fast scratch %p %lu\n",
        handles.cmd_data,
        static_cast<printf_size_t>(handles.cmd_data_size),
        handles.weight_data,
        static_cast<printf_size_t>(handles.weight_data_size),
        ethosu_scratch,
        static_cast<printf_size_t>(handles.scratch_data_size),
        ethosu_fast_scratch,
        static_cast<printf_size_t>(ethosu_fast_scratch_size));

    // Write argument values (from EValue tensor) into Ethos-U scratch
    // TODO(MLETORCH-123): Optimise into direct write from Vela into the SRAM
    //                     or DRAM output for compatible data layouts.
    for (int i = 0; i < input_count; i++) {
      auto tensor_count = 1, io_count = 1;
      auto tensor_in = args[i]->toTensor();

      // We accept:
      bool supported = 0;
      // 32 bit int (simple non-quantised test cases)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Int &&
           handles.inputs->io[i].elem_size == 4);
      // 8 bit int (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Char &&
           handles.inputs->io[i].elem_size == 1);
      // 8 bit uint8 (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Byte &&
           handles.inputs->io[i].elem_size == 1);
      // 16 bit int (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Short &&
           handles.inputs->io[i].elem_size == 2);
      // bool (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Bool &&
           handles.inputs->io[i].elem_size == 1);
      if (!supported) {
        ET_LOG(
            Error,
            "Input %d expected Integer (4 byte), Char (1 byte) or Bool (1 byte) integer inputs, got ScalarType id %s size %d",
            i,
            executorch::runtime::toString(tensor_in.scalar_type()),
            handles.inputs->io[i].elem_size);
        return Error::InvalidProgram;
      }

      if (needs_scratch_allocation()) {
        char* scratch_addr = ethosu_scratch + handles.inputs->io[i].offset;

        // Select a compatible copy routine including checking for input layouts
        // which require permutation.
        bool both_int = tensor_in.scalar_type() == ScalarType::Int &&
            handles.inputs->io[i].elem_size == 4;
        bool both_char = (tensor_in.scalar_type() == ScalarType::Char ||
                          tensor_in.scalar_type() == ScalarType::Byte) &&
            handles.inputs->io[i].elem_size == 1;
        bool both_short = tensor_in.scalar_type() == ScalarType::Short &&
            handles.inputs->io[i].elem_size == 2;
        bool both_bool = tensor_in.scalar_type() == ScalarType::Bool &&
            (handles.inputs->io[i].elem_size == 1);

        if (both_char || both_int || both_short || both_bool) {
          EXECUTORCH_PROF_SCOPE(
              event_tracer, "+EthosUBackend::execute()handles.input.memcpy()");
          // Sizes match and elt size matches so memcpy.
          // Routed through arm_ethos_io_memcpy so firmware can DMA-accelerate.
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
          EthosUBackend_input_memcpy(tensor_in.nbytes());
#endif
          arm_ethos_io_memcpy(
              scratch_addr,
              tensor_in.mutable_data_ptr<char>(),
              tensor_in.nbytes());
        } else {
          ET_LOG(Error, "No matching input copy routine");
          return Error::InvalidProgram;
        }
      }
      calculate_dimensions(
          tensor_in, &handles.inputs->io[i], &tensor_count, &io_count);
      if (tensor_count != io_count) {
        ET_LOG(Error, "Input tensor sizes do not match");
        ET_LOG(
            Error,
            "Program expects %d elements but got %d",
            io_count,
            tensor_count);
        return Error::InvalidProgram;
      }
    }

    EXECUTORCH_PROF_START(
        event_tracer, event_tracer_local_scope, "+EthosUBackend::execute()NPU");
    Error platform_status = platform_execute(
        context,
        execution_handle,
        handles,
        input_count,
        output_count,
        args,
        ethosu_scratch);
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);
    return platform_status;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle == nullptr) {
      return;
    }

    // Explicitly destroy platform-specific state before releasing the
    // execution handle to avoid leaking resources such as std::string.
    auto* exec_handle = reinterpret_cast<ExecutionHandle*>(handle);

    if (exec_handle->platform_state != nullptr) {
      platform_destroy(exec_handle->platform_state);
    }

    exec_handle->~ExecutionHandle();
  }

 private:
  // No platform-specific members.
};

void calculate_dimensions(
    const executorch::aten::Tensor tensor,
    VelaIO* io,
    int* tensor_count,
    int* io_count) {
  for (int i = 0; i < tensor.dim(); i++) {
    *tensor_count = *tensor_count * tensor.size(i);
  }

  // The VelaIO type has a shape of fixed size 6
  for (int i = 0; i < shapeDim; i++) {
    *io_count = *io_count * io->shape[i];
  }
}

namespace {
auto EthosUBackend_backend = EthosUBackend();
Backend EthosUBackend_id{"EthosUBackend", &EthosUBackend_backend};
static executorch::runtime::Error EthosUBackend_registered =
    register_backend(EthosUBackend_id);

} // namespace

} // namespace arm
} // namespace backends
} // namespace executorch
