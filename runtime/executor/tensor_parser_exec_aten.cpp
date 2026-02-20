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
namespace ET_RUNTIME_NAMESPACE {
namespace deserialization {

using executorch::aten::ScalarType;
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
  if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
    memory_offset |= static_cast<size_t>(memory_offset_high) << 32;
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        memory_offset_high == 0,
        NotSupported,
        "size_t cannot hold memory offset 0x%08" PRIx32 "%08" PRIx32,
        memory_offset_high,
        memory_offset_low);
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
        tensor_index >= 0 && static_cast<size_t>(tensor_index) < values_len,
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

ET_NODISCARD Error validateTensorLayout(
    const executorch_flatbuffer::Tensor* s_tensor,
    const TensorLayout& expected_layout) {
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<ScalarType>(s_tensor->scalar_type()) ==
          expected_layout.scalar_type(),
      InvalidExternalData,
      "Scalar type mismatch. Expected %hhd, got %hhd.",
      static_cast<int8_t>(s_tensor->scalar_type()),
      static_cast<int8_t>(expected_layout.scalar_type()));
  int dim = s_tensor->sizes()->size();
  ET_CHECK_OR_RETURN_ERROR(
      dim >= 0, InvalidExternalData, "Dim is negative: %d", dim)
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<size_t>(dim) == expected_layout.sizes().size(),
      InvalidExternalData,
      "Dim mismatch. Expected %d, got %zu.",
      dim,
      expected_layout.sizes().size());
  for (int i = 0; i < dim; i++) {
    ET_CHECK_OR_RETURN_ERROR(
        s_tensor->sizes()->Get(i) == expected_layout.sizes()[i],
        InvalidExternalData,
        "Sizes mismatch. Expected %d, got %d for size at index %d.",
        s_tensor->sizes()->Get(i),
        expected_layout.sizes()[i],
        i);
    ET_CHECK_OR_RETURN_ERROR(
        s_tensor->dim_order()->Get(i) == expected_layout.dim_order()[i],
        InvalidExternalData,
        "Dim order mismatch. Expected %d, got %d for dim at index %d.",
        s_tensor->dim_order()->Get(i),
        expected_layout.dim_order()[i],
        i);
  }
  return Error::Ok;
}

// Check if key exists in entries. If it does, return a pointer to the entry
// otherwise return a nullptr.
NamedData* get_data_by_key(const char* key, Span<NamedData> entries) {
  for (const auto i : c10::irange(entries.size())) {
    if (strcmp(key, entries[i].key) == 0) {
      return &entries[i];
    }
  }
  return nullptr;
}

ET_NODISCARD Result<void*> getTensorDataPtr(
    const executorch_flatbuffer::Tensor* s_tensor,
    const Program* program,
    size_t nbytes,
    HierarchicalAllocator* allocator,
    const NamedDataMap* named_data_map,
    Span<NamedData> external_constants) {
  auto data_buffer_idx = s_tensor->data_buffer_idx();
  const executorch_flatbuffer::AllocationDetails* allocation_info =
      s_tensor->allocation_info();

  // External tensors.
  if (s_tensor->extra_tensor_info() != nullptr &&
      s_tensor->extra_tensor_info()->location() ==
          executorch_flatbuffer::TensorDataLocation::EXTERNAL) {
    // Check that fqn is not null.
    ET_CHECK_OR_RETURN_ERROR(
        s_tensor->extra_tensor_info()->fully_qualified_name() != nullptr,
        InvalidExternalData,
        "Fully qualified name of external tensor is null");
    const char* fqn =
        s_tensor->extra_tensor_info()->fully_qualified_name()->c_str();

    // Constant value.
    if (allocation_info == nullptr) {
      NamedData* data = get_data_by_key(fqn, external_constants);
      if (data != nullptr) {
        return const_cast<void*>(data->buffer.data());
      }
      // Should never reach here; these tensors are resolved in
      // Method::parse_external_constants. Any errors should be caught there.
      return Error::Internal;
    } else {
      // Mutable value.
      // Look up tensor in named data map.
      ET_CHECK_OR_RETURN_ERROR(
          named_data_map != nullptr,
          InvalidExternalData,
          "Cannot retrieve external tensor with fqn: %s. The named_data_map is null; most likely no external .ptd file was provided.",
          fqn);
      Result<const TensorLayout> tensor_layout_res =
          named_data_map->get_tensor_layout(fqn);
      if (!tensor_layout_res.ok()) {
        return tensor_layout_res.error();
      }
      const TensorLayout& tensor_layout = tensor_layout_res.get();
      Error err = validateTensorLayout(s_tensor, tensor_layout);
      if (err != Error::Ok) {
        return err;
      }
      // Call load_into.
      auto planned_ptr = getMemPlannedPtr(allocation_info, nbytes, allocator);
      if (!planned_ptr.ok()) {
        return planned_ptr.error();
      }
      auto load_error =
          named_data_map->load_data_into(fqn, planned_ptr.get(), nbytes);
      if (load_error != Error::Ok) {
        return load_error;
      }

      return planned_ptr;
    }

    // Constant, stored in PTE file.
  } else if (data_buffer_idx > 0 && allocation_info == nullptr) {
    auto const_data =
        program->get_constant_buffer_data(data_buffer_idx, nbytes);
    if (!const_data.ok()) {
      return const_data.error();
    }

    // The const_cast is 'ok' here because the program and runtime should
    // guarantee that this data is never modified.
    return const_cast<void*>(const_data.get());

    // Memory Planned, with initial state
  } else if (data_buffer_idx > 0 && allocation_info != nullptr) {
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

    // Memory planned, no initial state
  } else if (data_buffer_idx == 0 && allocation_info != nullptr) {
    return getMemPlannedPtr(allocation_info, nbytes, allocator);

    // Pointer recived at runtime
  } else { // data_buffer_idx == 0 && allocation_info == nullptr,
    return nullptr;
  }
}

} // namespace deserialization
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
