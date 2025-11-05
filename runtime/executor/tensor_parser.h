/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __GNUC__
// Disable -Wdeprecated-declarations, as some builds use 'Werror'.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/schema/program_generated.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace deserialization {

/// Data structure to hold key and data buffer for external data used
/// in a method.
struct NamedData {
  const char* key;
  FreeableBuffer buffer;
};

NamedData* get_data_by_key(const char* key, Span<NamedData> entries);

ET_NODISCARD Result<executorch::aten::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor,
    const NamedDataMap* named_data_map = nullptr,
    Span<NamedData> external_constants = {});

ET_NODISCARD Result<BoxedEvalueList<executorch::aten::Tensor>> parseTensorList(
    const flatbuffers::Vector<int32_t>* tensor_indices,
    EValue* values,
    size_t values_len,
    MemoryManager* memory_manager);

// Checks that the sizes, dim_order and scalar_type match between tensors
// stored in the PTE and externally.
ET_NODISCARD Error validateTensorLayout(
    const executorch_flatbuffer::Tensor* s_tensor,
    const TensorLayout& expected_layout);

// Deserializes a List of optional type. The code here is the same between all
// list of optionals: list of optional Tensor, list of optional float etc, so we
// just use a template to avoid boilerplate.
template <typename T>
ET_NODISCARD Result<BoxedEvalueList<std::optional<T>>> parseListOptionalType(
    const flatbuffers::Vector<int32_t>* value_indices,
    EValue* values,
    size_t values_len,
    MemoryManager* memory_manager) {
  auto* evalp_list = memory_manager->method_allocator()->allocateList<EValue*>(
      value_indices->size());
  if (evalp_list == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  auto* optional_tensor_list =
      memory_manager->method_allocator()->allocateList<std::optional<T>>(
          value_indices->size());
  if (optional_tensor_list == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  size_t output_idx = 0;
  // For each index look up the corresponding EValue (which has been
  // already allocated) and stick it in the list.
  for (int32_t index : *value_indices) {
    // Lists of objects are stored in fbb as list[int] where the ints are
    // indices into values. Currently serialization is deciding if they want to
    // put -1 for serialized None type indices, or give us a valid index to a
    // serialized None. We support either for now.
    // Placement new as the list elements are not initialized, so calling
    // copy assignment is not defined if its non trivial.
    if (index == -1) {
      new (&optional_tensor_list[output_idx]) std::optional<T>(std::nullopt);
      // no value to point to. BoxedEvalueList for optional tensor will convert
      // this to nullopt.
      // TODO(T161156879): do something less hacky here.
      evalp_list[output_idx] = nullptr;
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          index >= 0 && static_cast<size_t>(index) < values_len,
          InvalidProgram,
          "Invalid value index %" PRId32 " for ListOptional",
          index);
      new (&optional_tensor_list[output_idx])
          std::optional<T>(values[index].toOptional<T>());
      evalp_list[output_idx] = &values[static_cast<size_t>(index)];
    }
    output_idx++;
  }
  return BoxedEvalueList<std::optional<T>>(
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
 * @param[in] named_data_map An optional map of {name, blob} used to resolve
 *     data that is mutable and external to the PTE, if any.
 * @param[in] external_constants An optional span containing tensor fqn to
 *     corresponding tensor data. Used to resolve data that is constant and
 *     external to the PTE, if any. Referencing data from external_constants is
 *     safe, as it has the same lifetime as the method.
 *
 * @returns On success, the data pointer to use for the tensor. On failure, a
 *     non-Ok Error.
 */
ET_NODISCARD Result<void*> getTensorDataPtr(
    const executorch_flatbuffer::Tensor* s_tensor,
    const Program* program,
    size_t nbytes,
    HierarchicalAllocator* allocator,
    const NamedDataMap* named_data_map = nullptr,
    Span<NamedData> external_constants = {});

} // namespace deserialization
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
namespace deserialization {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::deserialization::getTensorDataPtr;
using ::executorch::ET_RUNTIME_NAMESPACE::deserialization::
    parseListOptionalType;
using ::executorch::ET_RUNTIME_NAMESPACE::deserialization::parseTensor;
using ::executorch::ET_RUNTIME_NAMESPACE::deserialization::parseTensorList;
} // namespace deserialization
} // namespace executor
} // namespace torch

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
