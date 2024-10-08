/*
 * Copyright 2023 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Warning: Do not change this without changing arm_backend.py::vela_compile
 *          as that function emits this format and the two need to align.
 */

#include <executorch/backends/arm/runtime/VelaBinStream.h>

#include <cstring>

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace backends {
namespace arm {

// get next mul of 16 ptr, return n if already aligned
static uintptr_t next_mul_16(uintptr_t n) {
  return ((n - 1) | 15) + 1;
}

bool vela_bin_validate(const char* data, int size) {
  const char* foot = data + size - sizeof(VelaBinBlock);

  // Check 16 byte alignment
  bool valid = true;
  if ((uintptr_t)data != next_mul_16((uintptr_t)data)) {
    ET_LOG(Error, "Vela bin ptr not aligned to 16 bytes: %p", (void*)data);
    valid = false;
  }
  if ((uintptr_t)foot != next_mul_16((uintptr_t)foot)) {
    ET_LOG(Error, "End of vela bin not aligned to 16 bytes: %p", (void*)foot);
    valid = false;
  }
  // Check header and footer blocks are the right format
  if (strncmp(data, "vela_bin_stream", strlen("vela_bin_stream")) != 0) {
    ET_LOG(Error, "Incorrect header in vela_bin_stream");
    valid = false;
  }
  if (strncmp(foot, "vela_end_stream", strlen("vela_end_stream")) != 0) {
    ET_LOG(Error, "Incorrect footer in vela_bin_stream");
    valid = false;
  }

  return valid;
}

bool vela_bin_read(const char* data, VelaHandles* handles, int size) {
  const char* ptr = data;

  while (ptr - data < size) {
    VelaBinBlock* b = (VelaBinBlock*)ptr;
    ptr += sizeof(VelaBinBlock) + next_mul_16(b->size);

    if (!strncmp(b->name, "vela_bin_stream", strlen("vela_bin_stream"))) {
      // expect vela_bin_stream first
      if ((char*)b != (char*)data)
        return false;
    } else if (!strncmp(b->name, "cmd_data", strlen("cmd_data"))) {
      // This driver magic header confirms a valid command stream in binary
      if (strncmp(b->data, "COP1", strlen("COP1")))
        return false;
      handles->cmd_data = b->data;
      handles->cmd_data_size = b->size;
    } else if (!strncmp(b->name, "weight_data", strlen("weight_data"))) {
      handles->weight_data = b->data;
      handles->weight_data_size = b->size;
    } else if (!strncmp(b->name, "scratch_data", strlen("scratch_data"))) {
      handles->scratch_data = b->data;
      handles->scratch_data_size = b->size;
    } else if (!strncmp(b->name, "inputs", strlen("inputs"))) {
      handles->inputs = (VelaIOs*)b->data;
    } else if (!strncmp(b->name, "outputs", strlen("outputs"))) {
      handles->outputs = (VelaIOs*)b->data;
    } else if (!strncmp(
                   b->name, "vela_end_stream", strlen("vela_end_stream"))) {
      // expect vela_end_stream last
      if (ptr - data != size) {
        ET_LOG(Error, "Expected vela binary to end with vela_end_stream");
        return false;
      }
      return true;
    } else {
      // Unrecognised block name
      ET_LOG(Error, "Invalid block name or malformed binary");
      return false;
    }
  }

  // We've fallen off the end without finding vela_end_stream
  return false;
}

} // namespace arm
} // namespace backends
} // namespace executorch
