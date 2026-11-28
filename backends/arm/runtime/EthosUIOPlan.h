/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace arm {

enum class IOPlacement {
  INSIDE_SCRATCH,
  OUTSIDE_SCRATCH,
};

struct EthosUArguments {
  executorch::runtime::Span<executorch::runtime::EValue*> inputs;
  executorch::runtime::Span<executorch::runtime::EValue*> outputs;
};

struct IOPlan {
  bool needs_input_copy = false;
  bool needs_output_copy = false;
  IOPlacement placement = IOPlacement::INSIDE_SCRATCH;
  const VelaHandles* handles = nullptr;
  EthosUArguments arguments;
  executorch::runtime::Span<uint8_t> scratch;
  char* input_region_base = nullptr;
  size_t input_region_size = 0;
  char* contiguous_output_base = nullptr;
  size_t output_region_size = 0;
};

/**
 * Splits delegate arguments into inputs and outputs, and validates that all
 * arguments have usable tensor data.
 */
executorch::runtime::Result<EthosUArguments> parse_ethosu_arguments(
    const VelaHandles& handles,
    executorch::runtime::Span<executorch::runtime::EValue*> args);

/**
 * Creates the finalized I/O plan from Vela metadata, runtime tensors, resolved
 * scratch, and whether the platform supports direct I/O.
 */
executorch::runtime::Result<IOPlan> make_io_plan(
    const VelaHandles& handles,
    const EthosUArguments& arguments,
    executorch::runtime::Span<uint8_t> scratch,
    bool supports_direct_io);

size_t calculate_vela_io_bytes(const VelaIO& io);

executorch::runtime::Error copy_inputs(const IOPlan& io_plan);

executorch::runtime::Error copy_outputs(
    const IOPlan& io_plan,
    char* const* output_data = nullptr);

} // namespace arm
} // namespace backends
} // namespace executorch
