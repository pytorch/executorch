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
#include <pmu_ethosu.h>

#include "executorch/backends/arm/runtime/VelaBinStream.h"
#include "executorch/runtime/backend/interface.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/core/exec_aten/util/scalar_type_util.h"

using namespace std;

namespace torch {
namespace executor {

typedef struct {
  FreeableBuffer* processed;
  bool permuted_io_flag;
} ExecutionHandle;

class ArmBackend final : public PyTorchBackendInterface {
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
            toString(tensor_in.scalar_type()));
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
            scratch_addr,
            tensor_in.mutable_data_ptr<char>(),
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

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    return;
  }

 private:
  Error check_requires_permute(
      int index,
      const exec_aten::Tensor tensor_in,
      VelaIO* input,
      bool permuted_io_flag,
      bool* is_permuted) const {
    bool permuted_input_shape = false;
    if (tensor_in.dim() == 4) {
      // special case for NHWC workaround in AOT; as the compilation has
      // permuted to channel last in an undetectable way, we assume here
      // that the application has similarly permuted any input tensors.
      permuted_input_shape = tensor_in.size(0) == input->shape[0] &&
          tensor_in.size(1) == input->shape[3] &&
          tensor_in.size(2) == input->shape[1] &&
          tensor_in.size(3) == input->shape[2];
      if (permuted_input_shape) {
        ET_LOG(Info, "Tensor input %d will be permuted", index);
      }
      if (permuted_io_flag != permuted_input_shape) {
        ET_LOG(Error, "Permute compile flag and permuted input don't agree");
        return Error::InvalidProgram;
      }
    }
    if (!permuted_input_shape) {
      // Error check matching shapes in the general case
      for (int i = 0; i < tensor_in.dim(); i++) {
        if (tensor_in.size(i) != input->shape[i]) {
          ET_LOG(Error, "Tensor input %d mismatched shape", index);
          ET_LOG(
              Error,
              "dimension %d mismatch, %zd != %d",
              index,
              tensor_in.size(i),
              input->shape[i]);
          return Error::InvalidProgram;
        }
      }
    }
    *is_permuted = permuted_input_shape;
    return Error::Ok;
  }

  void permute_CHW_to_HWC(char* input, char* output, int H, int W) const {
    for (int i = 0; i != H * W; ++i) {
      output[i * 3 + 0] = input[i + 0 * W * H];
      output[i * 3 + 1] = input[i + 1 * W * H];
      output[i * 3 + 2] = input[i + 2 * W * H];
    }
  }
};

namespace {
auto backend = ArmBackend();
Backend backend_id{"ArmBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace executor
} // namespace torch
