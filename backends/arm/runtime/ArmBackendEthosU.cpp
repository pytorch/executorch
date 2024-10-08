/*
 * Copyright 2023-2024 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstring>
#include <memory>

#include <ethosu_driver.h>

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

namespace executorch {
namespace backends {
namespace arm {

typedef struct {
  FreeableBuffer* processed;
  bool permuted_io_flag;
} ExecutionHandle;

extern "C" {
void __attribute__((weak)) ArmBackend_execute_begin() {}
void __attribute__((weak)) ArmBackend_execute_end() {}
}

class ArmBackendExecuteCallbacks {
 public:
  ArmBackendExecuteCallbacks() {
    ArmBackend_execute_begin();
  }
  ~ArmBackendExecuteCallbacks() {
    ArmBackend_execute_end();
  }
};

class ArmBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ArmBackend() {}

  ~ArmBackend() = default;

  virtual bool is_available() const override {
    // TODO: revise to use a register check/init function
    return 1;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ET_LOG(Info, "ArmBackend::init %p", processed->data());

    char* data = (char*)processed->data();
    size_t size = processed->size();

    // Verify format of vela_bin
    if (vela_bin_validate(data, size) == false) {
      ET_LOG(Error, "Malformed vela_bin_stream found");
      return Error::InvalidProgram;
    }

    MemoryAllocator* allocator = context.get_runtime_allocator();
    ExecutionHandle* handle =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(allocator, ExecutionHandle);
    handle->processed = processed;

    handle->permuted_io_flag = false;
    for (auto& compile_spec : compile_specs) {
      if (0 == std::strcmp(compile_spec.key, "permute_memory_format") &&
          0 == std::memcmp(compile_spec.value.buffer, "nhwc", 4)) {
        handle->permuted_io_flag = true;
      }
    }

    // Return the same buffer we were passed - this data will be
    // executed directly
    return handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      EValue** args) const override {
    ExecutionHandle* execution_handle = (ExecutionHandle*)input_handle;
    VelaHandles handles;

    ArmBackendExecuteCallbacks ArmBackend_execute_callbacks;
    // Command stream - we know at this point it's aligned
    char* data = (char*)execution_handle->processed->data();
    ET_LOG(Info, "ArmBackend::execute %p", data);

    // Read key sections from the vela_bin_stream
    if (vela_bin_read(data, &handles, execution_handle->processed->size()) ==
        false) {
      ET_LOG(Error, "ArmBackend::vela_read: error, invalid binary layout");
      return Error::InvalidProgram;
    }

    ET_LOG(
        Debug,
        "ArmBackend::execute: Running program data:\n  cmd %p %zu\n  weight %p %zu\n  scratch %p %zu\n",
        handles.cmd_data,
        handles.cmd_data_size,
        handles.weight_data,
        handles.weight_data_size,
        handles.scratch_data,
        handles.scratch_data_size);

    // Write argument values (from EValue tensor) into Ethos-U scratch
    // TODO(MLETORCH-123): Optimise into direct write from Vela into the SRAM
    //                     or DRAM output for compatible data layouts.
    for (int i = 0; i < handles.inputs->count; i++) {
      auto tensor_in = args[i]->toTensor();
      char* scratch_addr = handles.scratch_data + handles.inputs->io[i].offset;

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
      if (!supported) {
        ET_LOG(
            Error,
            "Input %d expected Integer (4 byte) or Char (1 byte) integer inputs, got ScalarType id %s",
            i,
            executorch::runtime::toString(tensor_in.scalar_type()));
        return Error::InvalidProgram;
      }
      supported = executorch::runtime::is_contiguous_dim_order(
          tensor_in.dim_order().data(), tensor_in.dim());
      if (!supported) {
        ET_LOG(
            Error,
            "Input %d expected contiguous dim_order, but got non-contiguous dim_order",
            i);
        return Error::InvalidProgram;
      }

      // Select a compatible copy routine including checking for input layouts
      // which require permutation.
      bool permuted_input_shape;
      ET_CHECK_OK_OR_RETURN_ERROR(check_requires_permute(
          i,
          tensor_in,
          &handles.inputs->io[i],
          execution_handle->permuted_io_flag,
          &permuted_input_shape));
      bool both_char = tensor_in.scalar_type() == ScalarType::Char and
          handles.inputs->io[i].elem_size == 1;
      bool both_int = tensor_in.scalar_type() == ScalarType::Int and
          handles.inputs->io[i].elem_size == 4;

      // Select a compatible copy routine
      if (both_char and permuted_input_shape) {
        // permuted byte copy CHW to HWC
        permute_CHW_to_HWC(
            tensor_in.mutable_data_ptr<char>(),
            scratch_addr,
            tensor_in.size(1),
            tensor_in.size(2),
            tensor_in.size(3));
      } else if (both_char or both_int) {
        // Sizes match and elt size matches so memcpy
        memcpy(
            scratch_addr,
            tensor_in.mutable_data_ptr<char>(),
            tensor_in.nbytes());
      } else {
        ET_LOG(Error, "No matching input copy routine");
        return Error::InvalidProgram;
      }
    }

    // Allocate driver handle and synchronously invoke driver
    auto driver =
        std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
            ethosu_reserve_driver(), ethosu_release_driver);
    if (driver == NULL) {
      ET_LOG(Error, "ArmBackend::execute: ethosu_reserve_driver failed");
      return Error::InvalidState;
    }

    // Ethos-U low level driver expected order for Ethos U-55, we have
    // constant weight data, then scratch (which contains input and output)
    // scratch is written above in this function.
    uint64_t bases[2] = {
        (uint64_t)handles.weight_data, (uint64_t)handles.scratch_data};
    size_t bases_size[2] = {
        handles.weight_data_size, handles.scratch_data_size};
    int result = ethosu_invoke_v3(
        driver.get(),
        (void*)handles.cmd_data,
        handles.cmd_data_size,
        bases,
        bases_size,
        2, /* fixed array of pointers to binary interface*/
        nullptr);

    if (result != 0) {
      ET_LOG(
          Error,
          "ArmBackend::execute: Ethos-U invocation failed error (%d)",
          result);
      return Error::InvalidProgram;
    }

    // Write outputs from scratch into EValue pointers
    for (int i = 0; i < handles.outputs->count; i++) {
      const char* output_addr =
          handles.scratch_data + handles.outputs->io[i].offset;
      // Process input EValue into scratch
      // Outputs are in the index immediately after inputs
      auto tensor_out = args[handles.inputs->count + i]->toTensor();
      bool permuted_output_shape;
      ET_CHECK_OK_OR_RETURN_ERROR(check_requires_permute(
          i,
          tensor_out,
          &handles.outputs->io[i],
          execution_handle->permuted_io_flag,
          &permuted_output_shape));
      if (tensor_out.scalar_type() == ScalarType::Char and
          permuted_output_shape) {
        char* output_address = (char*)output_addr;
        permute_HWC_to_CHW(
            output_address,
            tensor_out.mutable_data_ptr<char>(),
            tensor_out.size(1),
            tensor_out.size(2),
            tensor_out.size(3));
      } else {
        for (int j = 0; j < tensor_out.numel(); j++) {
          if (tensor_out.scalar_type() == ScalarType::Char) {
            char* output_address = (char*)output_addr;
            tensor_out.mutable_data_ptr<char>()[j] = output_address[j];
          } else {
            int* output_address = (int*)output_addr;
            tensor_out.mutable_data_ptr<int>()[j] = output_address[j];
          }
        }
      }
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    return;
  }

 private:
  Error check_requires_permute(
      int index,
      const executorch::aten::Tensor tensor,
      VelaIO* io,
      bool permuted_io_flag,
      bool* is_permuted) const {
    bool permuted_shape = false;
    if (tensor.dim() == 4) {
      // special case for NHWC workaround in AOT; as the compilation has
      // permuted to channel last in an undetectable way, we assume here
      // that the application has similarly permuted any input/output tensors.
      permuted_shape = tensor.size(0) == io->shape[0] &&
          tensor.size(1) == io->shape[3] && tensor.size(2) == io->shape[1] &&
          tensor.size(3) == io->shape[2];
      if (permuted_shape) {
        ET_LOG(Info, "Tensor input/output %d will be permuted", index);
      }
      if (permuted_io_flag != permuted_shape) {
        ET_LOG(
            Error,
            "Permute compile flag and permuted input/output don't agree");
        return Error::InvalidProgram;
      }
    }
    if (!permuted_shape) {
      // Check the number of elements in each tensor match
      int tensor_count = 1;
      int io_count = 1;

      for (int i = 0; i < tensor.dim(); i++) {
        tensor_count = tensor_count * tensor.size(i);
      }

      // The VelaIO type has a shape of fixed size 4
      for (int i = 0; i < 4; i++) {
        io_count = io_count * io->shape[i];
      }

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
    *is_permuted = permuted_shape;
    return Error::Ok;
  }

  void permute_CHW_to_HWC(char* input, char* output, int C, int H, int W)
      const {
    for (int i = 0; i != H * W; ++i) {
      for (int j = 0; j < C; ++j) {
        output[i * C + j] = input[i + j * W * H];
      }
    }
  }

  void permute_HWC_to_CHW(char* input, char* output, int C, int H, int W)
      const {
    for (int i = 0; i != H * W; ++i) {
      for (int j = 0; j < C; ++j) {
        output[i + j * W * H] = input[i * C + j];
      }
    }
  }
};

namespace {
auto backend = ArmBackend();
Backend backend_id{"ArmBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace arm
} // namespace backends
} // namespace executorch
