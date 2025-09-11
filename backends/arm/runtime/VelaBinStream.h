/*
 * Copyright 2023-2024 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Minimal reading function for vela_bin_stream wire format. This is an
 * implementation detail of the arm_backend AoT flow and ArmBackendEthosU
 * and subject to change.
 * This format captures the command stream, I/O and memory layout data to
 * enable execution of the command stream on Ethos-U hardware.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace arm {

// Standard block name size
const uint32_t kVelaBlockNameLength = 16;

// Generic block within the vela_bin_stream encoded by the python vela_compile
// step
typedef struct {
  char name[kVelaBlockNameLength]; // string name, can be shorter or truncated
  uint32_t size; // unpadded size, BinBlock size will be rounded to next_mul_16
  char _pad[12]; // Our data often need 16 byte alignemnt
  char data[]; // block.name specific format data
} VelaBinBlock;

// A Vela input or output descriptor in the binary stream
typedef struct {
  int shape[4]; // Up to 4D shape of input or output
  int elem_size; // Element sizeof in bytes
  int offset; // Offset in bytes within SRAM working data
  int region; // Scratch region this belongs to
} VelaIO;

// A list of VelaIOs from the binary stream
typedef struct {
  int count;
  VelaIO io[];
} VelaIOs;

// Processed data used by the backend to invoke the payload
typedef struct {
  const char* cmd_data;
  size_t cmd_data_size;
  const char* weight_data;
  size_t weight_data_size;
  char* scratch_data;
  size_t scratch_data_size;
  VelaIOs* inputs;
  VelaIOs* outputs;
} VelaHandles;

/* Takes in the preprocessed vela_bin_stream wire format and returns data
 * needed to launch the workload on the Ethos-U and wire up input and
 * output values.
 */
bool vela_bin_read(const char* data, VelaHandles* handles, int size);

/* Does minimal validation of a vela_bin_stream to ensure the overall
 * structure is correct and so likely to contain valid binary data for launch
 * on the Ethos-U.
 */
bool vela_bin_validate(const char* data, int size);

} // namespace arm
} // namespace backends
} // namespace executorch
