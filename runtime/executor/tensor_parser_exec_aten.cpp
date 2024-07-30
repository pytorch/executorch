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

namespace torch {
namespace executor {
namespace deserialization {

__ET_NODISCARD Result<BoxedEvalueList<exec_aten::Tensor>> parseTensorList(
    const flatbuffers::Vector<int32_t>* tensor_indices,
    EValue* values_,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensorList");

  auto* tensor_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->method_allocator(),
      exec_aten::Tensor,
      tensor_indices->size());
  auto* evalp_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->method_allocator(), EValue*, tensor_indices->size());

  // For each tensor index look up the corresponding Tensor (which has been
  // already allocated) and stick it in the list.
  size_t output_idx = 0;
  for (int32_t tensor_index : *tensor_indices) {
    // Placement new as the list elements are not initialized, so calling
    // copy assignment is not defined if its non trivial.
    new (&tensor_list[output_idx]) exec_aten::Tensor(
        values_[static_cast<size_t>(tensor_index)].toTensor());
    evalp_list[output_idx] = &values_[static_cast<size_t>(tensor_index)];
    output_idx++;
  }

  return BoxedEvalueList<exec_aten::Tensor>(
      evalp_list, tensor_list, tensor_indices->size());
}

__ET_NODISCARD Result<void*> getTensorDataPtr(
    const executorch_flatbuffer::Tensor* s_tensor,
    const Program* program,
    size_t nbytes,
    HierarchicalAllocator* allocator) {
  if (s_tensor->data_buffer_idx() > 0) {
    auto data =
        program->get_constant_buffer_data(s_tensor->data_buffer_idx(), nbytes);
    if (!data.ok()) {
      return data.error();
    }
    // The const_cast is 'ok' here because the program and runtime should
    // guarantee that this data is never modified.
    return const_cast<void*>(data.get());
  }

  const executorch_flatbuffer::AllocationDetails* allocation_info =
      s_tensor->allocation_info();
  if (allocation_info != nullptr) {
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

  // The tensor's data will be allocated as part of execution.
  return nullptr;
}

} // namespace deserialization
} // namespace executor
} // namespace torch
