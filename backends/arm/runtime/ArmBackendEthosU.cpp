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
#include <memory>
#include <vector>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

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
    char* foot = data + size - 16;

    // Header and footer both 16 bit aligned suggest valid structure and we
    // wont walk off the end of the chunks and segfault
    if (!((int)data == next_mul_16((uintptr_t)data))) {
      ET_LOG(Error, "ArmBackend::init: Binary needs to be 16 byte unaligned");
      return Error::InvalidProgram;
    }
    if (!((int)foot == next_mul_16((uintptr_t)foot))) {
      ET_LOG(Error, "ArmBackend::init: Footer expected to be 16 byte aligned");
      ET_LOG(
          Error,
          "ArmBackend::init: Program expected to be multiple of 16 bytes");
      return Error::InvalidProgram;
    }
    if (!(0 == strncmp(data, "vela_bin_stream", 15))) {
      ET_LOG(Error, "ArmBackend::init: Binary passed is not a vela_bin_stream");
      return Error::InvalidProgram;
    }
    if (!(0 == strncmp(foot, "vela_end_stream", 15))) {
      ET_LOG(Error, "ArmBackend::init: Binary passed missing vela_end_stream");
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
    if (!this->vela_read(data, &handles, processed->size())) {
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
    for (int i = 0; i < handles.input_shapes.size(); i++) {
      const char* input_addr = handles.scratch_data + handles.input_offset[i];
      // Process input EValue into scratch
      // TODO: optimise into direct write for compatible, contig layout
      int* input_address = (int*)input_addr;
      auto tensor_in = args[i]->toTensor();
      for (int j = 0; j < tensor_in.numel(); j++) {
        // TODO: extend beyond 4 byte tensors
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

    // output data from Ethos U
    // We only handle one output at the moment
    const char* output_addr = handles.scratch_data + handles.output_offset[0];
    // Outputs are in the index immediately after inputs
    int output_index = handles.input_shapes.size();

    // Process results into EValue storage
    // TODO: optimise into direct write for compatible, contig layout
    int* output_address = (int*)output_addr;
    auto tensor_out = args[output_index]->toTensor();
    for (int j = 0; j < tensor_out.numel(); j++) {
      // TODO: extend beyond 4 byte tensors
      tensor_out.mutable_data_ptr<int>()[j] = output_address[j];
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    return;
  }

 private:
  typedef struct {
    const char* cmd_data;
    size_t cmd_data_size;
    const char* weight_data;
    size_t weight_data_size;
    const char* scratch_data;
    size_t scratch_data_size;
    vector<size_t> input_offset;
    vector<vector<int>> input_shapes;
    vector<size_t> output_offset;
    vector<vector<int>> output_shapes;
  } VelaHandles;

  typedef struct {
    char name[16];
    uint32_t size;
    char _pad[12];
    char data[];
  } VelaBinBlock;

  typedef struct {
    int count;
    int shape[][4];
  } VelaShapes;

  typedef struct {
    int count;
    int offsets[];
  } VelaOffsets;

  static int next_mul_16(int n) {
    return ((n - 1) | 15) + 1;
  }

  int vela_read(char* data, VelaHandles* handles, int size) const {
    constexpr const size_t header_size = 16;

    // Read header string
    if (strncmp(data, "vela_bin_stream", 15)) {
      return 0;
    }
    data += header_size;

    // Expect one or more 'VelaBinBlock's
    while (1) {
      VelaBinBlock* b = (VelaBinBlock*)data;
      data += sizeof(VelaBinBlock) + next_mul_16(b->size);

      // Exit with success on finding end of stream
      if (!strncmp(b->name, "vela_end_stream", strlen("vela_end_stream")))
        return 1;

      if (!strncmp(b->name, "cmd_data", strlen("cmd_data"))) {
        // This magic header confirms a valid command stream in binary
        if (strncmp(b->data, "COP1", strlen("COP1")))
          return 0;
        handles->cmd_data = b->data;
        handles->cmd_data_size = b->size;
      }
      if (!strncmp(b->name, "weight_data", strlen("weight_data"))) {
        handles->weight_data = b->data;
        handles->weight_data_size = b->size;
      }
      if (!strncmp(b->name, "scratch_data", strlen("scratch_data"))) {
        handles->scratch_data = b->data;
        handles->scratch_data_size = b->size;
      }

      // capture inputs and outputs
      if (!strncmp(b->name, "input_offset", strlen("input_offset"))) {
        VelaOffsets* offsets = (VelaOffsets*)b->data;
        for (int i = 0; i < offsets->count; i++) {
          handles->input_offset.push_back(offsets->offsets[i]);
        }
      }
      if (!strncmp(b->name, "output_offset", strlen("output_offset"))) {
        VelaOffsets* offsets = (VelaOffsets*)b->data;
        for (int i = 0; i < offsets->count; i++) {
          handles->output_offset.push_back(offsets->offsets[i]);
        }
      }

      if (!strncmp(b->name, "input_shape", strlen("input_shape"))) {
        VelaShapes* shapes = (VelaShapes*)b->data;
        for (int i = 0; i < shapes->count; i++) {
          vector<int> s = {
              shapes->shape[i][0],
              shapes->shape[i][1],
              shapes->shape[i][2],
              shapes->shape[i][3]};
          handles->input_shapes.push_back(s);
        }
      }
      if (!strncmp(b->name, "output_shape", strlen("output_shape"))) {
        VelaShapes* shapes = (VelaShapes*)b->data;
        for (int i = 0; i < shapes->count; i++) {
          vector<int> s = {
              shapes->shape[i][0],
              shapes->shape[i][1],
              shapes->shape[i][2],
              shapes->shape[i][3]};
          handles->output_shapes.push_back(s);
        }
      }
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
