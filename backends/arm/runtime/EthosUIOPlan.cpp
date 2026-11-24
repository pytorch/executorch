/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <limits>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/EthosUIOPlan.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::Span;

// Firmware targets can override the default memcpy implementation to route
// I/O copies through a DMA engine.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size);

namespace executorch {
namespace backends {
namespace arm {

constexpr int ETHOSU_INPUT_REGION = 3;
constexpr int ETHOSU_OUTPUT_REGION = 4;
constexpr int ETHOSU_SCRATCH_REGION = 1;

size_t calculate_vela_io_bytes(const VelaIO& io) {
  size_t elements = 1;
  for (int dim = 0; dim < shapeDim; ++dim) {
    const int dimension = io.shape[dim];
    elements *= static_cast<size_t>(dimension == 0 ? 1 : dimension);
  }
  return elements * static_cast<size_t>(io.elem_size);
}

namespace {

struct OutputCopyLayout {
  size_t chunk_count;
  size_t vela_chunk_size;
  size_t tensor_chunk_size;

  OutputCopyLayout(size_t tensor_bytes, const VelaIO& output_io)
      : chunk_count(1), vela_chunk_size(0), tensor_chunk_size(0) {
    for (int dim = 0; dim < shapeDim - 1; ++dim) {
      const int dimension = output_io.shape[dim];
      chunk_count *= static_cast<size_t>(dimension == 0 ? 1 : dimension);
    }
    const int last_dimension = output_io.shape[shapeDim - 1];
    vela_chunk_size =
        static_cast<size_t>(last_dimension == 0 ? 1 : last_dimension) *
        static_cast<size_t>(output_io.elem_size);
    tensor_chunk_size = tensor_bytes / chunk_count;
  }
};

Error validate_output_copy_layout(
    const executorch::aten::Tensor& tensor,
    const VelaIO& output_io,
    int output_index) {
  if (output_io.elem_size <= 0) {
    ET_LOG(
        Error,
        "Ethos-U output %d reports invalid element size %d",
        output_index,
        output_io.elem_size);
    return Error::InvalidProgram;
  }
  if (output_io.elem_size != tensor.element_size()) {
    ET_LOG(
        Error,
        "Ethos-U output %d element size mismatch: tensor %zu, Vela %d",
        output_index,
        tensor.element_size(),
        output_io.elem_size);
    return Error::InvalidProgram;
  }

  const size_t tensor_bytes = tensor.nbytes();
  const OutputCopyLayout layout(tensor_bytes, output_io);
  if (tensor_bytes % layout.chunk_count != 0) {
    ET_LOG(
        Error,
        "Ethos-U output %d tensor bytes %zu not divisible by chunk count %zu",
        output_index,
        tensor_bytes,
        layout.chunk_count);
    return Error::InvalidProgram;
  }

  if (layout.vela_chunk_size < layout.tensor_chunk_size) {
    ET_LOG(
        Error,
        "Ethos-U output %d Vela chunk bytes %zu are smaller than tensor chunk bytes %zu",
        output_index,
        layout.vela_chunk_size,
        layout.tensor_chunk_size);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

struct MemoryRegion {
  uintptr_t base = 0;
  size_t required_size = 0;
  bool is_contiguous = false;
  bool has_zero_offset = false;
};

bool is_cop2(const VelaHandles& handles) {
  return handles.cmd_data != nullptr && handles.cmd_data_size >= 4 &&
      std::strncmp(handles.cmd_data, "COP2", 4) == 0;
}

/** Determines whether Vela addresses I/O inside or outside scratch. */
Result<IOPlacement> get_io_placement(const VelaHandles& handles) {
  if (!is_cop2(handles)) {
    return IOPlacement::INSIDE_SCRATCH;
  }

  const int input_count = handles.inputs == nullptr ? 0 : handles.inputs->count;
  const int output_count =
      handles.outputs == nullptr ? 0 : handles.outputs->count;
  bool separate_regions = false;
  if (input_count > 0) {
    const int region = handles.inputs->io[0].region;
    if (region != ETHOSU_SCRATCH_REGION && region != ETHOSU_INPUT_REGION) {
      ET_LOG(Error, "Ethos-U input uses unsupported memory region %d", region);
      return Error::InvalidProgram;
    }
    separate_regions = region == ETHOSU_INPUT_REGION;
  } else if (output_count > 0) {
    const int region = handles.outputs->io[0].region;
    if (region != ETHOSU_SCRATCH_REGION && region != ETHOSU_OUTPUT_REGION) {
      ET_LOG(Error, "Ethos-U output uses unsupported memory region %d", region);
      return Error::InvalidProgram;
    }
    separate_regions = region == ETHOSU_OUTPUT_REGION;
  }

  const int expected_input_region =
      separate_regions ? ETHOSU_INPUT_REGION : ETHOSU_SCRATCH_REGION;
  for (int i = 0; i < input_count; ++i) {
    if (handles.inputs->io[i].region != expected_input_region) {
      ET_LOG(Error, "Ethos-U inputs use inconsistent memory regions");
      return Error::InvalidProgram;
    }
  }
  const int expected_output_region =
      separate_regions ? ETHOSU_OUTPUT_REGION : ETHOSU_SCRATCH_REGION;
  for (int i = 0; i < output_count; ++i) {
    if (handles.outputs->io[i].region != expected_output_region) {
      ET_LOG(Error, "Ethos-U inputs and outputs use inconsistent regions");
      return Error::InvalidProgram;
    }
  }
  return separate_regions ? IOPlacement::OUTSIDE_SCRATCH
                          : IOPlacement::INSIDE_SCRATCH;
}

bool input_pair_supported(
    const executorch::aten::Tensor& tensor_in,
    const VelaIO& input_io) {
  bool supported = false;
  supported |=
      tensor_in.scalar_type() == ScalarType::Int && input_io.elem_size == 4;
  supported |=
      tensor_in.scalar_type() == ScalarType::Char && input_io.elem_size == 1;
  supported |=
      tensor_in.scalar_type() == ScalarType::Byte && input_io.elem_size == 1;
  supported |=
      tensor_in.scalar_type() == ScalarType::Short && input_io.elem_size == 2;
  supported |=
      tensor_in.scalar_type() == ScalarType::Bool && input_io.elem_size == 1;
  return supported;
}

Error validate_inputs(const VelaIOs* inputs, Span<EValue*> args) {
  const int input_count = inputs == nullptr ? 0 : inputs->count;
  for (int i = 0; i < input_count; ++i) {
    const VelaIO& input = inputs->io[i];
    const auto tensor = args[i]->toTensor();
    if (!input_pair_supported(tensor, input)) {
      ET_LOG(
          Error,
          "Input %d EValue type %s is incompatible with Vela element size %d",
          i,
          executorch::runtime::toString(tensor.scalar_type()),
          input.elem_size);
      return Error::InvalidProgram;
    }

    const size_t input_bytes = calculate_vela_io_bytes(input);
    if (tensor.nbytes() != input_bytes) {
      ET_LOG(
          Error,
          "Input %d tensor bytes %zu do not match Ethos-U input bytes %zu",
          i,
          tensor.nbytes(),
          input_bytes);
      return Error::InvalidProgram;
    }
  }
  return Error::Ok;
}

/**
 * Validates all output copy layouts and reports whether every tensor has the
 * same byte size as its Vela output.
 */
Result<bool> validate_outputs(const VelaIOs* outputs, Span<EValue*> args) {
  const int output_count = outputs == nullptr ? 0 : outputs->count;
  bool tensor_sizes_match = true;
  for (int i = 0; i < output_count; ++i) {
    const VelaIO& output = outputs->io[i];
    auto tensor = args[i]->toTensor();
    Error status = validate_output_copy_layout(tensor, output, i);
    if (status != Error::Ok) {
      return status;
    }
    tensor_sizes_match &= tensor.nbytes() == calculate_vela_io_bytes(output);
  }
  return tensor_sizes_match;
}

/**
 * Derives the common region base, required size, and contiguity of runtime
 * tensors using their Vela offsets.
 */
Result<MemoryRegion> get_memory_region(const VelaIOs* ios, Span<EValue*> args) {
  const int io_count = ios == nullptr ? 0 : ios->count;
  MemoryRegion region;
  if (io_count == 0) {
    region.is_contiguous = true;
    return region;
  }

  uintptr_t region_base = 0;
  bool region_base_set = false;
  bool all_match = true;

  for (int i = 0; i < io_count; ++i) {
    const VelaIO& io = ios->io[i];
    const size_t io_bytes = calculate_vela_io_bytes(io);
    auto tensor = args[i]->toTensor();
    if (io.offset < 0) {
      ET_LOG(Error, "Ethos-U I/O %d has negative offset %d", i, io.offset);
      return Error::InvalidProgram;
    }
    const size_t offset = static_cast<size_t>(io.offset);
    if (io_bytes > std::numeric_limits<size_t>::max() - offset) {
      ET_LOG(Error, "Ethos-U I/O %d offset and size overflow", i);
      return Error::InvalidProgram;
    }
    const size_t end = offset + io_bytes;

    region.required_size = std::max(region.required_size, end);
    region.has_zero_offset |= offset == 0;

    const char* tensor_ptr = tensor.const_data_ptr<char>();
    bool pointer_matches = false;

    if (tensor_ptr != nullptr) {
      const uintptr_t address = reinterpret_cast<uintptr_t>(tensor_ptr);
      if (!region_base_set && address >= offset) {
        region_base = address - offset;
        region_base_set = true;
      }
      if (region_base_set &&
          offset <= std::numeric_limits<uintptr_t>::max() - region_base) {
        pointer_matches = address == region_base + offset;
      }
    }

    all_match &= pointer_matches;
  }

  region.base = region_base_set ? region_base : 0;
  region.is_contiguous = all_match;
  return region;
}

bool ranges_overlap(
    uintptr_t first,
    size_t first_size,
    uintptr_t second,
    size_t second_size) {
  if (first_size == 0 || second_size == 0) {
    return false;
  }
  return first <= second ? second - first < first_size
                         : first - second < second_size;
}

Error validate_input_copy_aliases(const IOPlan& io_plan) {
  const auto inputs = io_plan.arguments.inputs;
  const auto* input_ios = io_plan.handles->inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto tensor = inputs[i]->toTensor();
    const size_t bytes = tensor.nbytes();
    if (bytes == 0) {
      continue;
    }
    if (io_plan.input_region_base == nullptr) {
      ET_LOG(Error, "Ethos-U input copy received null buffer");
      return Error::InvalidState;
    }
    const uintptr_t source =
        reinterpret_cast<uintptr_t>(tensor.const_data_ptr<char>());
    const uintptr_t destination =
        reinterpret_cast<uintptr_t>(io_plan.input_region_base) +
        input_ios->io[i].offset;
    if (source == destination) {
      continue;
    }
    for (size_t j = 0; j < inputs.size(); ++j) {
      const auto other = inputs[j]->toTensor();
      const size_t other_bytes = other.nbytes();
      const uintptr_t other_source =
          reinterpret_cast<uintptr_t>(other.const_data_ptr<char>());
      if (ranges_overlap(destination, bytes, other_source, other_bytes)) {
        ET_LOG(Error, "Ethos-U input copy would overwrite input %zu", j);
        return Error::InvalidProgram;
      }
      if (other_bytes > 0 &&
          input_ios->io[i].offset == input_ios->io[j].offset &&
          (source != other_source || bytes != other_bytes)) {
        ET_LOG(
            Error, "Ethos-U duplicate input destination has different sources");
        return Error::InvalidProgram;
      }
    }
  }
  return Error::Ok;
}

} // namespace

Result<EthosUArguments> parse_ethosu_arguments(
    const VelaHandles& handles,
    Span<EValue*> args) {
  const size_t input_count = static_cast<size_t>(
      handles.inputs == nullptr ? 0 : handles.inputs->count);
  const size_t output_count = static_cast<size_t>(
      handles.outputs == nullptr ? 0 : handles.outputs->count);
  const size_t io_argument_count = input_count + output_count;
  if ((handles.inputs != nullptr && handles.inputs->count < 0) ||
      (handles.outputs != nullptr && handles.outputs->count < 0) ||
      args.size() != io_argument_count) {
    ET_LOG(Error, "Invalid Ethos-U argument count");
    return Error::InvalidProgram;
  }

  EthosUArguments arguments;
  arguments.inputs =
      Span<EValue*>(input_count == 0 ? nullptr : args.data(), input_count);
  arguments.outputs = Span<EValue*>(
      output_count == 0 ? nullptr : args.data() + args.size() - output_count,
      output_count);

  for (size_t i = 0; i < input_count; ++i) {
    if (arguments.inputs[i] == nullptr || !arguments.inputs[i]->isTensor()) {
      ET_LOG(Error, "Ethos-U input argument %zu is not a tensor", i);
      return Error::InvalidProgram;
    }
    auto tensor = arguments.inputs[i]->toTensor();
    if (tensor.nbytes() > 0 && tensor.const_data_ptr<char>() == nullptr) {
      ET_LOG(Error, "Ethos-U input argument %zu has no data", i);
      return Error::InvalidProgram;
    }
  }
  for (size_t i = 0; i < output_count; ++i) {
    if (arguments.outputs[i] == nullptr || !arguments.outputs[i]->isTensor()) {
      ET_LOG(Error, "Ethos-U output argument %zu is not a tensor", i);
      return Error::InvalidProgram;
    }
    auto tensor = arguments.outputs[i]->toTensor();
    if (tensor.nbytes() > 0 && tensor.const_data_ptr<char>() == nullptr) {
      ET_LOG(Error, "Ethos-U output argument %zu has no data", i);
      return Error::InvalidProgram;
    }
  }
  return arguments;
}

Result<IOPlan> make_io_plan(
    const VelaHandles& handles,
    const EthosUArguments& arguments,
    Span<uint8_t> scratch,
    bool supports_direct_io) {
  auto io_placement = get_io_placement(handles);
  if (!io_placement.ok()) {
    return io_placement.error();
  }
  const int input_count = handles.inputs == nullptr ? 0 : handles.inputs->count;
  const int output_count =
      handles.outputs == nullptr ? 0 : handles.outputs->count;

  Error status = validate_inputs(handles.inputs, arguments.inputs);
  if (status != Error::Ok) {
    return status;
  }
  auto output_tensor_sizes_match =
      validate_outputs(handles.outputs, arguments.outputs);
  if (!output_tensor_sizes_match.ok()) {
    return output_tensor_sizes_match.error();
  }

  auto input_region = get_memory_region(handles.inputs, arguments.inputs);
  if (!input_region.ok()) {
    return input_region.error();
  }
  auto output_region = get_memory_region(handles.outputs, arguments.outputs);
  if (!output_region.ok()) {
    return output_region.error();
  }
  const bool separate_regions = *io_placement == IOPlacement::OUTSIDE_SCRATCH;
  if (supports_direct_io && !separate_regions &&
      (input_region->required_size > handles.scratch_data_size ||
       output_region->required_size > handles.scratch_data_size)) {
    ET_LOG(Error, "Ethos-U I/O exceeds scratch buffer bounds");
    return Error::InvalidProgram;
  }
  if (supports_direct_io && !separate_regions &&
      (input_count > 0 || output_count > 0) &&
      (scratch.data() == nullptr ||
       scratch.size() < handles.scratch_data_size)) {
    ET_LOG(Error, "Ethos-U scratch allocation is missing or too small");
    return Error::InvalidState;
  }

  const char* scratch_base = reinterpret_cast<const char*>(scratch.data());
  bool inputs_can_be_direct = false;
  bool outputs_can_be_direct = false;
  if (supports_direct_io) {
    if (separate_regions) {
      inputs_can_be_direct = input_region->is_contiguous &&
          (input_count == 0 || input_region->has_zero_offset);
      outputs_can_be_direct = *output_tensor_sizes_match &&
          output_region->is_contiguous &&
          (output_count == 0 || output_region->has_zero_offset);
    } else {
      inputs_can_be_direct = input_region->is_contiguous &&
          input_region->base == reinterpret_cast<uintptr_t>(scratch_base);
      outputs_can_be_direct = *output_tensor_sizes_match &&
          output_region->is_contiguous &&
          output_region->base == reinterpret_cast<uintptr_t>(scratch_base);
    }
  }
  inputs_can_be_direct |= input_count == 0;
  outputs_can_be_direct |= output_count == 0;

  IOPlan io_plan;
  io_plan.placement = *io_placement;
  io_plan.handles = &handles;
  io_plan.arguments = arguments;
  io_plan.scratch = scratch;
  io_plan.input_region_base = inputs_can_be_direct
      ? reinterpret_cast<char*>(input_region->base)
      : nullptr;
  io_plan.input_region_size = input_region->required_size;
  io_plan.contiguous_output_base = outputs_can_be_direct
      ? reinterpret_cast<char*>(output_region->base)
      : nullptr;
  io_plan.output_region_size = output_region->required_size;
  io_plan.needs_input_copy = !inputs_can_be_direct;
  io_plan.needs_output_copy = !outputs_can_be_direct;
  return io_plan;
}

Error copy_inputs(const IOPlan& io_plan) {
  if (!io_plan.needs_input_copy) {
    return Error::Ok;
  }
  // Validate every copy before any destination can overwrite an input source.
  const Error status = validate_input_copy_aliases(io_plan);
  if (status != Error::Ok) {
    return status;
  }
  const VelaHandles& handles = *io_plan.handles;
  Span<EValue*> input_args = io_plan.arguments.inputs;
  const int input_count = handles.inputs == nullptr ? 0 : handles.inputs->count;
  for (int i = 0; i < input_count; ++i) {
    auto tensor = input_args[i]->toTensor();
    if (tensor.nbytes() == 0) {
      continue;
    }
    char* dest = io_plan.input_region_base + handles.inputs->io[i].offset;
    if (dest != tensor.const_data_ptr<char>()) {
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
      EthosUBackend_input_memcpy(tensor.nbytes());
#endif
      arm_ethos_io_memcpy(dest, tensor.const_data_ptr<char>(), tensor.nbytes());
    }
  }
  return Error::Ok;
}

Error copy_outputs(const IOPlan& io_plan, char* const* output_data) {
  const VelaHandles& handles = *io_plan.handles;
  Span<EValue*> output_args = io_plan.arguments.outputs;
  const int output_count =
      handles.outputs == nullptr ? 0 : handles.outputs->count;
  if (!io_plan.needs_output_copy || output_count == 0) {
    return Error::Ok;
  }
  if (io_plan.contiguous_output_base == nullptr && output_data == nullptr) {
    ET_LOG(Error, "Ethos-U output copy received no output buffers");
    return Error::InvalidState;
  }
  for (int output_index = 0; output_index < output_count; ++output_index) {
    const VelaIO& output_io = handles.outputs->io[output_index];
    const char* output_buffer = output_data == nullptr
        ? (io_plan.contiguous_output_base == nullptr
               ? nullptr
               : io_plan.contiguous_output_base + output_io.offset)
        : output_data[output_index];
    auto tensor_out = output_args[output_index]->toTensor();
    const size_t tensor_bytes = tensor_out.nbytes();
    if (tensor_bytes == 0) {
      continue;
    }
    if (output_buffer == nullptr) {
      ET_LOG(Error, "Ethos-U output copy received null buffer");
      return Error::InvalidState;
    }
    if (output_buffer ==
        static_cast<const char*>(tensor_out.const_data_ptr())) {
      continue;
    }
    const OutputCopyLayout layout(tensor_bytes, output_io);

    char* dest = tensor_out.mutable_data_ptr<char>();
    if (dest == nullptr) {
      ET_LOG(Error, "Ethos-U padded copy received null destination");
      return Error::InvalidState;
    }
    const char* src_bytes = output_buffer;
    for (size_t chunk_idx = 0; chunk_idx < layout.chunk_count; ++chunk_idx) {
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
      EthosUBackend_output_memcpy(layout.tensor_chunk_size);
#endif
      arm_ethos_io_memcpy(dest, src_bytes, layout.tensor_chunk_size);
      src_bytes += layout.vela_chunk_size;
      dest += layout.tensor_chunk_size;
    }
  }
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
