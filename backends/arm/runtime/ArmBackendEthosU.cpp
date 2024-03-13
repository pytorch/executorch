/*
 * Copyright 2023 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstring>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <executorch/backends/arm/runtime/VelaBinStream.h>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

using namespace std;

namespace torch {
namespace executor {

// TODO: we should be in 0x31, to access a full 2MB SRAM
// region and enable maximum program performance up to
// 2MB, rather than 1.
// SRAM (rwx) : ORIGIN = 0x31000000, LENGTH = 0x00200000
#define CS300_SRAM_LOW ((void*)0x11000000)
#define CS300_SRAM_HIGH ((void*)0x110FFFFF)

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
    char* foot = data + size - sizeof(VelaBinBlock);

    // Verify format of vela_bin
    if (vela_bin_validate(data, size) == false) {
      ET_LOG(Error, "Malformed vela_bin_stream found");
      return Error::InvalidProgram;
    }

    // Verify address range is accessible current expectation is the program
    // is wholly stored in SRAM
    // TODO: expect to improve capabilities here by supporting DRAM storage
    //       and only moving required data into SRAM.
    if (!(data > CS300_SRAM_LOW || foot < CS300_SRAM_HIGH)) {
      ET_LOG(Error, "ArmBackend::init: Expected program binary to be in SRAM");
      ET_LOG(
          Error,
          "ArmBackend::init: program binary range %p:%p",
          data,
          foot + 16);
      return Error::InvalidProgram;
    }

    // Return the same buffer we were passed - this data will be
    // executed directly
    return processed;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      EValue** args) const override {
    FreeableBuffer* processed = (FreeableBuffer*)input_handle;

    ET_LOG(Info, "ArmBackend::execute %p", processed->data());

    VelaHandles handles;

    // Command stream - we know at this point it's aligned
    char* data = (char*)processed->data();

    // Read key sections from the vela_bin_stream
    if (vela_bin_read(data, &handles, processed->size()) == false) {
      ET_LOG(Error, "ArmBackend::vela_read: error, invalid binary layout");
      return Error::InvalidProgram;
    }

    ET_LOG(
        Debug,
        "ArmBackend::execute: Running program data:\n  cmd %p %d\n  weight %p %d\n  scratch %p %d\n",
        handles.cmd_data,
        handles.cmd_data_size,
        handles.weight_data,
        handles.weight_data_size,
        handles.scratch_data,
        handles.scratch_data_size);

    // Write inputs into SRAM scratch area defined by Vela
    for (int i = 0; i < handles.inputs->count; i++) {
      const char* input_addr =
          handles.scratch_data + handles.inputs->io[i].offset;
      // Process input EValue into scratch
      // TODO: Optimise into direct write from Vela into the SRAM or DRAM output
      //       for compatible data layouts.
      int* input_address = (int*)input_addr;
      auto tensor_in = args[i]->toTensor();
      for (int j = 0; j < tensor_in.numel(); j++) {
        // TODO: extend beyond tensors with 4 byte elements
        input_address[j] = tensor_in.mutable_data_ptr<int>()[j];
      }
    }

    // Allocate driver handle and synchronously invoke driver
    ethosu_driver* drv = ethosu_reserve_driver();
    if (drv == NULL) {
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
        drv,
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
      int* output_address = (int*)output_addr;
      // Outputs are in the index immediately after inputs
      auto tensor_out = args[handles.inputs->count + i]->toTensor();
      for (int j = 0; j < tensor_out.numel(); j++) {
        tensor_out.mutable_data_ptr<int>()[j] = output_address[j];
      }
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    return;
  }
};

namespace {
auto backend = ArmBackend();
Backend backend_id{"ArmBackend", &backend};
static auto registered = register_backend(backend_id);
} // namespace

} // namespace executor
} // namespace torch
