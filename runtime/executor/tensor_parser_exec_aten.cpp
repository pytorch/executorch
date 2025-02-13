/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

namespace executorch {
namespace runtime {
namespace deserialization {

using executorch::aten::ScalarType;
using executorch::runtime::TensorLayout;
// Provides access to private Program methods.
class TensorParser final {
 public:
  ET_NODISCARD static Error load_mutable_subsegment_into(
      const Program* program,
      size_t mutable_data_segments_index,
      size_t offset_index,
      size_t size,
      void* buffer) {
    return program->load_mutable_subsegment_into(
        mutable_data_segments_index, offset_index, size, buffer);
  }
};

namespace {

// Retrieve the buffer specified by the allocation_info
ET_NODISCARD Result<void*> getMemPlannedPtr(
    const executorch_flatbuffer::AllocationDetails* allocation_info,
    size_t nbytes,
    HierarchicalAllocator* allocator) {
  // Normal non-constant Tensor. Allocate data using mem_id and offset.

  // TODO(T142455629): make the allocator actually id based and not indexed
  // based. -1 is a hack to get the memory ids 0 aligned because previously
  // 0 was reserved
  const uint32_t memory_id = allocation_info->memory_id() - 1;

  // Originally this field was a single uint32_t, but we need 64 bits for
  // larger models. To preserve backwards compatibility, the high bits are
  // managed in a separate uint32_t field.
  const uint32_t memory_offset_low = allocation_info->memory_offset_low();
  const uint32_t memory_offset_high = allocation_info->memory_offset_high();

  size_t memory_offset = memory_offset_low;
  if (memory_offset_high > 0) {
    // The compiler should remove this always-true check on 64-bit systems.
    ET_CHECK_OR_RETURN_ERROR(
        sizeof(size_t) >= sizeof(uint64_t),
        NotSupported,
        "size_t cannot hold memory offset 0x%08" PRIx32 ".%08" PRIx32,
        memory_offset_high,
        memory_offset_low);
    memory_offset |= static_cast<size_t>(memory_offset_high) << 32;
  }
  return allocator->get_offset_address(memory_id, memory_offset, nbytes);
}
} // namespace

ET_NODISCARD Result<BoxedEvalueList<executorch::aten::Tensor>> parseTensorList(
    const flatbuffers::Vector<int32_t>* tensor_indices,
    EValue* values,
    size_t values_len,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensorList");

  auto* tensor_list =
      memory_manager->method_allocator()
          ->allocateList<executorch::aten::Tensor>(tensor_indices->size());
  if (tensor_list == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  auto* evalp_list = memory_manager->method_allocator()->allocateList<EValue*>(
      tensor_indices->size());
  if (evalp_list == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // For each tensor index look up the corresponding Tensor (which has been
  // already allocated) and stick it in the list.
  size_t output_idx = 0;
  for (int32_t tensor_index : *tensor_indices) {
    ET_CHECK_OR_RETURN_ERROR(
        tensor_index >= 0 && tensor_index < values_len,
        InvalidProgram,
        "Invalid value index %" PRId32 " for TensorList",
        tensor_index);

    // Placement new as the list elements are not initialized, so calling
    // copy assignment is not defined if it's non trivial.
    new (&tensor_list[output_idx]) executorch::aten::Tensor(
        values[static_cast<size_t>(tensor_index)].toTensor());
    evalp_list[output_idx] = &values[static_cast<size_t>(tensor_index)];
    output_idx++;
  }

  return BoxedEvalueList<executorch::aten::Tensor>(
      evalp_list, tensor_list, tensor_indices->size());
}

ET_NODISCARD Result<void*> getTensorDataPtr(
    const executorch_flatbuffer::Tensor* s_tensor,
    const Program* program,
    size_t nbytes,
    HierarchicalAllocator* allocator,
    const NamedDataMap* named_data_map) {
  auto data_buffer_idx = s_tensor->data_buffer_idx();
  const executorch_flatbuffer::AllocationDetails* allocation_info =
      s_tensor->allocation_info();

  // Memory Planned, with initial state
  if (data_buffer_idx > 0 && allocation_info != nullptr) {
    auto planned_ptr = getMemPlannedPtr(allocation_info, nbytes, allocator);
    if (!planned_ptr.ok()) {
      return planned_ptr.error();
    }
    auto err = TensorParser::load_mutable_subsegment_into(
        program, 0, s_tensor->data_buffer_idx(), nbytes, planned_ptr.get());

    if (err != Error::Ok) {
      return err;
    }
    return planned_ptr;
  }

  // External tensors.
  else if (
      s_tensor->extra_tensor_info() != nullptr &&
      s_tensor->extra_tensor_info()->location() ==
          executorch_flatbuffer::TensorDataLocation::EXTERNAL) {
    // Check that fqn is not null.
    ET_CHECK_OR_RETURN_ERROR(
        s_tensor->extra_tensor_info()->fully_qualified_name() != nullptr,
        InvalidExternalData,
        "Fully qualified name of external tensor is null");
    // Look up tensor in named data map.
    Result<const TensorLayout> tensor_layout_res = named_data_map->get_metadata(
        s_tensor->extra_tensor_info()->fully_qualified_name()->c_str());
    if (!tensor_layout_res.ok()) {
      return tensor_layout_res.error();
    }
    const TensorLayout& tensor_layout = tensor_layout_res.get();

    // Compatibility checking.
    ET_CHECK_OR_RETURN_ERROR(
        static_cast<ScalarType>(s_tensor->scalar_type()) ==
            tensor_layout.scalar_type(),
        InvalidExternalData,
        "Scalar type mismatch. Expected %hhd, got %hhd.",
        static_cast<int8_t>(s_tensor->scalar_type()),
        static_cast<int8_t>(tensor_layout.scalar_type()));
    ET_CHECK_OR_RETURN_ERROR(
        nbytes == tensor_layout.nbytes(),
        InvalidExternalData,
        "Nbytes mismatch. Expected %zu, got %zu.",
        nbytes,
        tensor_layout.nbytes());
    int dim = s_tensor->sizes()->size();
    ET_CHECK_OR_RETURN_ERROR(
        dim == tensor_layout.sizes().size(),
        InvalidExternalData,
        "Dim mismatch. Expected %d, got %zu.",
        dim,
        tensor_layout.sizes().size());
    for (int i = 0; i < dim; i++) {
      ET_CHECK_OR_RETURN_ERROR(
          s_tensor->sizes()->Get(i) == tensor_layout.sizes()[i],
          InvalidExternalData,
          "Sizes mismatch. Expected %d, got %d for size at index %d.",
          s_tensor->sizes()->Get(i),
          tensor_layout.sizes()[i],
          i);
      ET_CHECK_OR_RETURN_ERROR(
          s_tensor->dim_order()->Get(i) == tensor_layout.dim_order()[i],
          InvalidExternalData,
          "Dim order mismatch. Expected %d, got %d for dim at index %d.",
          s_tensor->dim_order()->Get(i),
          tensor_layout.dim_order()[i],
          i);
    }

    // Constant value.
    if (allocation_info == nullptr) {
      Result<FreeableBuffer> data_res = named_data_map->get_data(
          s_tensor->extra_tensor_info()->fully_qualified_name()->c_str());
      if (!data_res.ok()) {
        return data_res.error();
      }
      // The const_cast is 'ok' here because program and runtime should
      // guarantee that this data is never modified. Temporary until runtime
      // takes ownership of FreeableBuffers in TODO(T214294528).
      return const_cast<void*>(data_res.get().data());
    }

    // Mutable value.
    else {
      // Call load_into.
      auto planned_ptr = getMemPlannedPtr(allocation_info, nbytes, allocator);
      if (!planned_ptr.ok()) {
        return planned_ptr.error();
      }
      auto size = named_data_map->load_data_into(
          s_tensor->extra_tensor_info()->fully_qualified_name()->c_str(),
          planned_ptr.get(),
          nbytes);
      if (size.error() != Error::Ok) {
        return size.error();
      }
      ET_CHECK_OR_RETURN_ERROR(
          size.get() == nbytes,
          InvalidExternalData,
          "Expected to load %zu bytes, actually loaded %u bytes",
          nbytes,
          static_cast<unsigned int>(size.get()));
      return planned_ptr;
    }
  }

  // Constant, stored in PTE file.
  else if (data_buffer_idx > 0 && allocation_info == nullptr) {
    auto const_data =
        program->get_constant_buffer_data(data_buffer_idx, nbytes);
    if (!const_data.ok()) {
      return const_data.error();
    }

    // The const_cast is 'ok' here because the program and runtime should
    // guarantee that this data is never modified.
    return const_cast<void*>(const_data.get());

    // Memory planned, no initial state
  } else if (data_buffer_idx == 0 && allocation_info != nullptr) {
    return getMemPlannedPtr(allocation_info, nbytes, allocator);

    // Pointer recived at runtime
  } else { // data_buffer_idx == 0 && allocation_info == nullptr,
    return nullptr;
  }
}

} // namespace deserialization
} // namespace runtime
} // namespace executorch
