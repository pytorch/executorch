/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/program_validation.h>

#include <cstdint>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/program_generated.h>

#include <c10/util/safe_numerics.h>

namespace executorch {
namespace runtime {

ET_NODISCARD Error
validate_tensor(const executorch_flatbuffer::Tensor* tensor) {
  if (tensor == nullptr) {
    return Error::InvalidProgram;
  }

  const auto* sizes = tensor->sizes();
  if (sizes == nullptr) {
    return Error::InvalidProgram;
  }

  ssize_t numel = 1;
  for (flatbuffers::uoffset_t i = 0; i < sizes->size(); i++) {
    int32_t size = sizes->Get(i);

    if (size < 0) {
      ET_LOG(
          Error,
          "Size must be non-negative, got %d at dimension %u",
          size,
          static_cast<unsigned>(i));
      return Error::InvalidProgram;
    }

    bool overflow =
        c10::mul_overflows(numel, static_cast<ssize_t>(size), &numel);
    if (overflow) {
      ET_LOG(
          Error,
          "numel overflowed at dimension %u with size %d",
          static_cast<unsigned>(i),
          size);
      return Error::InvalidProgram;
    }
  }

  auto scalar_type =
      static_cast<executorch::aten::ScalarType>(tensor->scalar_type());
  if (!executorch::runtime::isValid(scalar_type)) {
    return Error::InvalidProgram;
  }

  size_t nbytes;
  bool nbytes_overflow = c10::mul_overflows(
      static_cast<size_t>(numel),
      executorch::runtime::elementSize(scalar_type),
      &nbytes);
  if (nbytes_overflow) {
    ET_LOG(
        Error,
        "nbytes overflowed: numel %zd with element size %zu",
        numel,
        executorch::runtime::elementSize(scalar_type));
    return Error::InvalidProgram;
  }

  return Error::Ok;
}

ET_NODISCARD Error
validate_program(const executorch_flatbuffer::Program* program) {
  if (program == nullptr) {
    return Error::InvalidProgram;
  }

  // Validate all execution plans.
  const auto* execution_plans = program->execution_plan();
  if (execution_plans == nullptr) {
    ET_LOG(Error, "Program has null execution_plan");
    return Error::InvalidProgram;
  }

  for (flatbuffers::uoffset_t plan_idx = 0; plan_idx < execution_plans->size();
       plan_idx++) {
    const auto* plan = execution_plans->Get(plan_idx);
    if (plan == nullptr) {
      ET_LOG(
          Error, "Execution plan %u is null", static_cast<unsigned>(plan_idx));
      return Error::InvalidProgram;
    }

    // Validate all values in the plan.
    const auto* values = plan->values();
    if (values == nullptr) {
      ET_LOG(
          Error,
          "Execution plan %u has null values table",
          static_cast<unsigned>(plan_idx));
      return Error::InvalidProgram;
    }

    for (flatbuffers::uoffset_t value_idx = 0; value_idx < values->size();
         value_idx++) {
      const auto* value = values->Get(value_idx);
      if (value == nullptr) {
        continue;
      }

      // Check if this value is a tensor.
      if (value->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
        const auto* tensor =
            static_cast<const executorch_flatbuffer::Tensor*>(value->val());

        Error err = validate_tensor(tensor);
        if (err != Error::Ok) {
          ET_LOG(
              Error,
              "Tensor validation failed for value %u in execution plan %u",
              static_cast<unsigned>(value_idx),
              static_cast<unsigned>(plan_idx));
          return err;
        }
      }

      // Check if this value is a TensorList.
      if (value->val_type() == executorch_flatbuffer::KernelTypes::TensorList) {
        const auto* tensor_list =
            static_cast<const executorch_flatbuffer::TensorList*>(value->val());

        if (tensor_list == nullptr) {
          ET_LOG(
              Error,
              "TensorList is null for value %u in execution plan %u",
              static_cast<unsigned>(value_idx),
              static_cast<unsigned>(plan_idx));
          return Error::InvalidProgram;
        }

        const auto* items = tensor_list->items();
        if (items == nullptr) {
          ET_LOG(Error, "TensorList items is null");
          return Error::InvalidProgram;
        }

        // Validate that each item index points to a Tensor evalue.
        for (flatbuffers::uoffset_t item_idx = 0; item_idx < items->size();
             item_idx++) {
          int32_t evalue_index = items->Get(item_idx);

          // Check bounds.
          if (evalue_index < 0 ||
              static_cast<flatbuffers::uoffset_t>(evalue_index) >=
                  values->size()) {
            ET_LOG(
                Error,
                "TensorList item %u has out-of-bounds index %d (values size "
                "%u) in execution plan %u",
                static_cast<unsigned>(item_idx),
                evalue_index,
                static_cast<unsigned>(values->size()),
                static_cast<unsigned>(plan_idx));
            return Error::InvalidProgram;
          }

          // Check that the referenced evalue is actually a Tensor.
          const auto* referenced_value = values->Get(evalue_index);
          if (referenced_value == nullptr) {
            ET_LOG(
                Error,
                "TensorList item %u references null evalue at index %d in "
                "execution plan %u",
                static_cast<unsigned>(item_idx),
                evalue_index,
                static_cast<unsigned>(plan_idx));
            return Error::InvalidProgram;
          }

          if (referenced_value->val_type() !=
              executorch_flatbuffer::KernelTypes::Tensor) {
            ET_LOG(
                Error,
                "TensorList item %u references non-Tensor evalue (type %d) at "
                "index %d in execution plan %u",
                static_cast<unsigned>(item_idx),
                static_cast<int>(referenced_value->val_type()),
                evalue_index,
                static_cast<unsigned>(plan_idx));
            return Error::InvalidProgram;
          }
        }
      }
    }
  }

  return Error::Ok;
}

} // namespace runtime
} // namespace executorch
