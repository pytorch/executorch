/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/schema/schema_generated.h>

namespace torch {
namespace executor {
namespace deserialization {

Result<exec_aten::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor);

inline Result<BoxedEvalueList<exec_aten::Tensor>> parseTensorList(
    const flatbuffers::Vector<int32_t>* tensor_indices,
    EValue* values_,
    MemoryManager* memory_manager) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensorList");

  auto* tensor_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->get_runtime_allocator(),
      exec_aten::Tensor,
      tensor_indices->size());
  auto* evalp_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->get_runtime_allocator(), EValue*, tensor_indices->size());

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

// Deserializes a List of optional type. The code here is the same between all
// list of optionals: list of optional Tensor, list of optional float etc, so we
// just use a template to avoid boilerplate.
template <typename T>
inline Result<BoxedEvalueList<exec_aten::optional<T>>> parseListOptionalType(
    const flatbuffers::Vector<int32_t>* value_indices,
    EValue* values_,
    MemoryManager* memory_manager) {
  auto* evalp_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->get_runtime_allocator(), EValue*, value_indices->size());

  auto* optional_tensor_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager->get_runtime_allocator(),
      exec_aten::optional<T>,
      value_indices->size());

  size_t output_idx = 0;
  // For each index look up the corresponding EValue (which has been
  // already allocated) and stick it in the list.
  for (int32_t index : *value_indices) {
    // Lists of objects are stored in fbb as list[int] where the ints are
    // indices into values_. Currently serialization is deciding if they want to
    // put -1 for serialized None type indices, or give us a valid index to a
    // serialized None. We support either for now.
    // Placement new as the list elements are not initialized, so calling
    // copy assignment is not defined if its non trivial.
    if (index == -1) {
      new (&optional_tensor_list[output_idx])
          exec_aten::optional<T>(exec_aten::nullopt);
    } else {
      new (&optional_tensor_list[output_idx])
          exec_aten::optional<T>(values_[index].toOptional<T>());
    }
    evalp_list[output_idx] = &values_[static_cast<size_t>(index)];
    output_idx++;
  }
  return BoxedEvalueList<exec_aten::optional<T>>(
      evalp_list, optional_tensor_list, value_indices->size());
}

/**
 * Returns the appropriate data pointer for `s_tensor`.
 *
 * Overall, a Tensor is either constant or non-constant, except we differentiate
 * 2 special variants of non-constant Tensor ("input" and control-flow
 * "placeholder") as a special optimization to avoid holding unnecessary
 * AllocationDetails. Thus, s_tensor can be configured as 1 of 3 options:
 * - constant_buffer > 0, allocation_info = Null: Constant Tensor.
 * - constant_buffer = 0, allocation_info = Non Null: Non-constant Tensor.
 * - constant_buffer = 0, allocation_info = Null: Input/placeholder Tensor.
 *
 * @param[in] s_tensor The tensor to find the data pointer for.
 * @param[in] program The Program to use for constant buffer data.
 * @param[in] nbytes The amount of memory to get from the allocator.
 * @param[in] allocator The source of memory for non-constant tensors.
 *
 * @returns On success, the data pointer to use for the tensor. On failure, a
 *     non-Ok Error.
 */
__ET_NODISCARD inline Result<void*> getTensorDataPtr(
    const executorch_flatbuffer::Tensor* s_tensor,
    const Program* program,
    size_t nbytes,
    HierarchicalAllocator* allocator) {
  if (s_tensor->constant_buffer_idx() > 0) {
    const void* data =
        program->get_constant_buffer_data(s_tensor->constant_buffer_idx());
    // The const_cast is ok here because the program and runtime should
    // guarantee that this data is never modified.
    return const_cast<void*>(data);
  }

  const executorch_flatbuffer::AllocationDetails* allocation_info =
      s_tensor->allocation_info();
  if (allocation_info != nullptr) {
    // Normal non-constant Tensor. Allocate data using mem_id and offset.

    // TODO(T142455629): make the allocator actually id based and not indexed
    // based. -1 is a hack to get the memory ids 0 aligned because previously
    // 0 was reserved
    const uint32_t memory_id = allocation_info->memory_id() - 1;
    return allocator->get_offset_address(
        memory_id, allocation_info->memory_offset(), nbytes);
  }

  // The tensor's data will be allocated as part of execution.
  return nullptr;
}

} // namespace deserialization
} // namespace executor
} // namespace torch
