/*
 * Copyright 2023-2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstdint>
#include <cstring>
#include <memory>

#include <ethosu_driver.h>

#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerEntry;

class EventTraceScope {
 public:
  EventTraceScope(EventTracer* event_tracer_, const char* name) {
    event_tracer = event_tracer_;
    event_tracer_entry_scope = event_tracer->start_profiling(name);
  }
  ~EventTraceScope() {
    event_tracer->end_profiling(event_tracer_entry_scope);
  }

 private:
  EventTracer* event_tracer;
  EventTracerEntry event_tracer_entry_scope;
};
#define EXECUTORCH_PROF_SCOPE(EVENTTRACER, NAME) \
  EventTraceScope event_tracer_scope = EventTraceScope(EVENTTRACER, NAME)
#define EXECUTORCH_PROF_START(EVENTTRACER, SCOPE, NAME) \
  SCOPE = EVENTTRACER->start_profiling(NAME)
#define EXECUTORCH_PROF_END(EVENTTRACER, SCOPE) \
  EVENTTRACER->end_profiling(SCOPE)

#else
#define EXECUTORCH_PROF_SCOPE(EVENTTRACER, NAME)
#define EXECUTORCH_PROF_START(EVENTTRACER, SCOPE, NAME)
#define EXECUTORCH_PROF_END(EVENTTRACER, SCOPE)
#endif

#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

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

#define ETHOSU_NUM_BASE_ADDRS 3

namespace executorch {
namespace backends {
namespace arm {

typedef struct {
  FreeableBuffer* processed;
} ExecutionHandle;

extern "C" {
void __attribute__((weak)) EthosUBackend_execute_begin() {}
void __attribute__((weak)) EthosUBackend_execute_end() {}
__attribute__((weak)) unsigned char* ethosu_fast_scratch = nullptr;
__attribute__((weak)) size_t ethosu_fast_scratch_size = 0;
}

class EthosUBackendExecuteCallbacks {
 public:
  EthosUBackendExecuteCallbacks() {
    EthosUBackend_execute_begin();
  }
  ~EthosUBackendExecuteCallbacks() {
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
    ET_LOG(Info, "data:%p", processed->data());

    const char* data = static_cast<const char*>(processed->data());
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

    handle->processed = processed;

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
    EthosUBackendExecuteCallbacks CollectArm_CPU_Cycles;

    ExecutionHandle* execution_handle =
        static_cast<ExecutionHandle*>(input_handle);
    VelaHandles handles;

    // Command stream - we know at this point it's aligned
    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::execute()processed_data");
    const char* data =
        static_cast<const char*>(execution_handle->processed->data());
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);

    ET_LOG(Debug, "data:%p", data);

    EXECUTORCH_PROF_START(
        event_tracer,
        event_tracer_local_scope,
        "+EthosUBackend::execute()vela_bin_read()");
    // Read key sections from the vela_bin_stream
    if (vela_bin_read(data, &handles, execution_handle->processed->size()) ==
        false) {
      ET_LOG(Error, "vela_read: error, invalid binary layout");
      return Error::InvalidProgram;
    }
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);

    MemoryAllocator* temp_allocator = context.get_temp_allocator();
    // Use a temporary allocator for the intermediate tensors of the
    // computation. The allocator is released in runtime/executor/method.cpp at
    // the end of the execution of the Ethos-U custom delegate
    // Ethos-U driver requires 16 bit alignment.
    char* ethosu_scratch = static_cast<char*>(
        temp_allocator->allocate(handles.scratch_data_size, 16UL));
    if (ethosu_scratch == nullptr) {
      ET_LOG(
          Error,
          "Failed to allocate scratch buffer of %zu bytes from temp_allocator",
          handles.scratch_data_size);
      return Error::MemoryAllocationFailed;
    }
    ET_LOG(
        Debug,
        "Running program data:\n  cmd %p %zu\n  weight %p %zu\n  scratch %p %zu\n  fast scratch %p %zu\n",
        handles.cmd_data,
        handles.cmd_data_size,
        handles.weight_data,
        handles.weight_data_size,
        ethosu_scratch,
        handles.scratch_data_size,
        ethosu_fast_scratch,
        ethosu_fast_scratch_size);

    // Write argument values (from EValue tensor) into Ethos-U scratch
    // TODO(MLETORCH-123): Optimise into direct write from Vela into the SRAM
    //                     or DRAM output for compatible data layouts.
    for (int i = 0; i < handles.inputs->count; i++) {
      auto tensor_count = 1, io_count = 1;
      auto tensor_in = args[i]->toTensor();
      char* scratch_addr = ethosu_scratch + handles.inputs->io[i].offset;

      // We accept:
      bool supported = 0;
      // 32 bit int (simple non-quantised test cases)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Int and
           handles.inputs->io[i].elem_size == 4);
      // 8 bit int (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Char and
           handles.inputs->io[i].elem_size == 1);
      // 16 bit int (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Short and
           handles.inputs->io[i].elem_size == 2);
      // bool (IOQDQ pass prepared networks)
      supported |=
          (tensor_in.scalar_type() == ScalarType::Bool and
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

      // Select a compatible copy routine including checking for input layouts
      // which require permutation.
      bool both_int = tensor_in.scalar_type() == ScalarType::Int &&
          handles.inputs->io[i].elem_size == 4;
      bool both_char = tensor_in.scalar_type() == ScalarType::Char &&
          handles.inputs->io[i].elem_size == 1;
      bool both_short = tensor_in.scalar_type() == ScalarType::Short &&
          handles.inputs->io[i].elem_size == 2;
      bool both_bool = tensor_in.scalar_type() == ScalarType::Bool &&
          (handles.inputs->io[i].elem_size == 1);

      if (both_char || both_int || both_short || both_bool) {
        EXECUTORCH_PROF_SCOPE(
            event_tracer, "+EthosUBackend::execute()handles.input.memcpy()");
        // Sizes match and elt size matches so memcpy
        memcpy(
            scratch_addr,
            tensor_in.mutable_data_ptr<char>(),
            tensor_in.nbytes());
      } else {
        ET_LOG(Error, "No matching input copy routine");
        return Error::InvalidProgram;
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

    // Allocate driver handle and synchronously invoke driver
    auto driver =
        std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
            ethosu_reserve_driver(), ethosu_release_driver);
    if (driver == NULL) {
      ET_LOG(Error, "ethosu_reserve_driver failed");
      return Error::InvalidState;
    }

    // Ethos-U low level driver expected order for Ethos U-55, we have
    // constant weight data, then scratch (which contains input and output)
    // scratch is written above in this function.

    uint64_t bases[ETHOSU_NUM_BASE_ADDRS] = {
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>((handles.weight_data))),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_scratch)),
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(ethosu_fast_scratch))};
    size_t bases_size[ETHOSU_NUM_BASE_ADDRS] = {
        handles.weight_data_size,
        handles.scratch_data_size,
        ethosu_fast_scratch_size};
    int result = 0;
    EXECUTORCH_PROF_START(
        event_tracer, event_tracer_local_scope, "+EthosUBackend::execute()NPU");
    result = ethosu_invoke_v3(
        driver.get(),
        static_cast<const void*>(handles.cmd_data),
        handles.cmd_data_size,
        bases,
        bases_size,
        ETHOSU_NUM_BASE_ADDRS, /* fixed array of pointers to binary interface*/
        nullptr);
    EXECUTORCH_PROF_END(event_tracer, event_tracer_local_scope);

    if (result != 0) {
      ET_LOG(Error, "Ethos-U invocation failed error (%d)", result);
      return Error::InvalidProgram;
    }
    size_t tensor_bytes_total = 0;
    size_t io_bytes_total = 0;
    // Write outputs from scratch into EValue pointers
    for (int i = 0; i < handles.outputs->count; i++) {
      int tensor_count = 1, io_count = 1;
      const char* output_addr = ethosu_scratch + handles.outputs->io[i].offset;
      // Process input EValue into scratch
      // Outputs are in the index immediately after inputs
      auto tensor_out = args[handles.inputs->count + i]->toTensor();

      calculate_dimensions(
          tensor_out, &handles.outputs->io[i], &tensor_count, &io_count);

      size_t tensor_bytes = tensor_out.nbytes();
      size_t io_bytes = static_cast<size_t>(io_count) *
          static_cast<size_t>(handles.outputs->io[i].elem_size);

      if (tensor_bytes != io_bytes) {
        Error status = copy_with_layout_adjustment(
            handles.outputs->io[i], i, output_addr, tensor_out, tensor_bytes);
        if (status != Error::Ok) {
          return status;
        }
        io_bytes_total += tensor_bytes;
      } else {
        EXECUTORCH_PROF_SCOPE(
            event_tracer, "+EthosUBackend::execute()handles.output.memcpy()");

        memcpy(
            tensor_out.mutable_data_ptr<char>(),
            static_cast<const char*>(output_addr),
            tensor_bytes);
        io_bytes_total += io_bytes;
      }

      // At times the topological order of the outputs may change.
      // Lets instead ensure that the sum of output bytes match.
      tensor_bytes_total += tensor_bytes;
    }
    if (tensor_bytes_total != io_bytes_total) {
      ET_LOG(Error, "Total output tensor sizes do not match");
      ET_LOG(
          Error,
          "Program expects %zu bytes but got %zu",
          io_bytes_total,
          tensor_bytes_total);
      return Error::InvalidProgram;
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    return;
  }

 private:
  // Copies Vela output into the ExecuTorch tensor, adjusting for padding or
  // packed layouts produced by the delegate.
  Error copy_with_layout_adjustment(
      const VelaIO& output_io,
      int output_index,
      const char* src,
      executorch::aten::Tensor& tensor_out,
      size_t tensor_bytes) const {
    const int elem_size = output_io.elem_size;
    if (elem_size == 0) {
      ET_LOG(
          Error, "Ethos-U output %d reports zero element size", output_index);
      return Error::InvalidProgram;
    }

    size_t chunk_count = 1;
    for (int dim = 0; dim < shapeDim - 1; ++dim) {
      const int vela_dim = output_io.shape[dim];
      chunk_count *= static_cast<size_t>(vela_dim == 0 ? 1 : vela_dim);
    }
    const int last_dim = output_io.shape[shapeDim - 1];
    const size_t vela_chunk_elems =
        static_cast<size_t>(last_dim == 0 ? 1 : last_dim);
    const size_t vela_chunk_size =
        vela_chunk_elems * static_cast<size_t>(elem_size);

    if (tensor_bytes % chunk_count != 0) {
      ET_LOG(
          Error,
          "Ethos-U output %d tensor bytes %zu not divisible by chunk count %zu",
          output_index,
          tensor_bytes,
          chunk_count);
      return Error::InvalidProgram;
    }

    const size_t chunk_size = tensor_bytes / chunk_count;

    // If Vela writes fewer bytes than the tensor expects we may need to
    // expand 4-bit data to 8-bit. Ethos-U outputs may be
    // packed 4-bit values but ExecuTorch tensors are at least 8-bit.
    if (vela_chunk_size < chunk_size) {
      if (chunk_size % vela_chunk_size != 0) {
        ET_LOG(
            Error,
            "Ethos-U output %d chunk bytes %zu not divisible by vela chunk bytes %zu",
            output_index,
            chunk_size,
            vela_chunk_size);
        return Error::InvalidProgram;
      }

      const size_t expand_factor = chunk_size / vela_chunk_size;
      if (expand_factor == 2 && elem_size == 1 &&
          tensor_out.scalar_type() == ScalarType::Char) {
        return unpack_chunks_4bit_to_int8(
            reinterpret_cast<const uint8_t*>(src),
            tensor_out.mutable_data_ptr<int8_t>(),
            chunk_count,
            chunk_size,
            vela_chunk_size);
      }

      ET_LOG(
          Error,
          "Ethos-U output %d expansion factor %zu with element size %d not supported",
          output_index,
          expand_factor,
          elem_size);
      return Error::InvalidProgram;
    }

    return strip_delegate_padding(
        src,
        tensor_out.mutable_data_ptr<char>(),
        chunk_count,
        chunk_size,
        vela_chunk_size);
  }

  Error unpack_chunks_4bit_to_int8(
      const uint8_t* src,
      int8_t* dest,
      size_t chunk_count,
      size_t dest_chunk_size,
      size_t src_chunk_size) const {
    const uint8_t* chunk_src = src;
    int8_t* chunk_dest = dest;
    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
      unpack_single_chunk_4bit_to_int8(chunk_src, chunk_dest, src_chunk_size);
      chunk_src += src_chunk_size;
      chunk_dest += dest_chunk_size;
    }
    return Error::Ok;
  }

  void unpack_single_chunk_4bit_to_int8(
      const uint8_t* src,
      int8_t* dest,
      size_t chunk_size) const {
    for (size_t byte_idx = 0; byte_idx < chunk_size; ++byte_idx) {
      const uint8_t packed = src[byte_idx];
      int8_t low = static_cast<int8_t>(packed & 0x0F);
      int8_t high = static_cast<int8_t>((packed >> 4) & 0x0F);
      if (low >= 8) {
        low -= 16;
      }
      if (high >= 8) {
        high -= 16;
      }
      dest[2 * byte_idx] = low;
      dest[2 * byte_idx + 1] = high;
    }
  }

  Error strip_delegate_padding(
      const char* src,
      char* dest,
      size_t chunk_count,
      size_t dest_chunk_size,
      size_t src_chunk_size) const {
    if (dest_chunk_size > src_chunk_size) {
      ET_LOG(
          Error,
          "dest chunk size %zu must not exceed src chunk size %zu",
          dest_chunk_size,
          src_chunk_size);
      return Error::InvalidProgram;
    }
    if (src == nullptr || dest == nullptr) {
      ET_LOG(Error, "Ethos-U padded copy received null buffer");
      return Error::InvalidState;
    }
    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
      memcpy(dest, src, dest_chunk_size);
      src += src_chunk_size;
      dest += dest_chunk_size;
    }
    return Error::Ok;
  }

  void calculate_dimensions(
      const executorch::aten::Tensor tensor,
      VelaIO* io,
      int* tensor_count,
      int* io_count) const {
    for (int i = 0; i < tensor.dim(); i++) {
      *tensor_count = *tensor_count * tensor.size(i);
    }

    // The VelaIO type has a shape of fixed size 6
    for (int i = 0; i < shapeDim; i++) {
      *io_count = *io_count * io->shape[i];
    }
  }
};

namespace {
auto backend = EthosUBackend();
Backend backend_id{"EthosUBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace arm
} // namespace backends
} // namespace executorch
