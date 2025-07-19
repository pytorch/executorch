/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/safe_numerics.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/schema/program_generated.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

namespace {
Result<Tag> get_tag(
    flatbuffers::Vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>::
        return_type serialization_value,
    size_t index) {
  switch (serialization_value->val_type()) {
    case executorch_flatbuffer::KernelTypes::Null: {
      return Tag::None;
    } break;
    case executorch_flatbuffer::KernelTypes::Int: {
      return Tag::Int;
    } break;
    case executorch_flatbuffer::KernelTypes::Double: {
      return Tag::Double;
    } break;
    case executorch_flatbuffer::KernelTypes::Bool: {
      return Tag::Bool;
    } break;
    case executorch_flatbuffer::KernelTypes::String: {
      return Tag::String;
    } break;
    case executorch_flatbuffer::KernelTypes::Tensor: {
      return Tag::Tensor;
    } break;
    default:
      ET_LOG(
          Error,
          "Invalid tag: %zu input idx: %zu",
          (size_t)serialization_value->val_type(),
          index);
      return Error::Internal;
  }
}

Result<size_t> calculate_nbytes(
    Span<const int32_t> sizes,
    executorch::aten::ScalarType scalar_type) {
  size_t n = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    size_t next_n;
    bool overflow =
        c10::mul_overflows(n, static_cast<size_t>(sizes[i]), &next_n);
    ET_CHECK_OR_RETURN_ERROR(
        !overflow,
        InvalidArgument,
        "Invalid size[%zu]: %d. Potentially overflowed, expect to be 0 or n: %zu",
        i,
        sizes[i],
        n);
    n = next_n;
  }

  size_t elem_size = executorch::runtime::elementSize(scalar_type);
  size_t total_bytes;
  bool overflow = c10::mul_overflows(n, elem_size, &total_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      !overflow,
      InvalidArgument,
      "Invalid elem_size: %zu. Potentially overflowed, expect to be 0 or n: %zu",
      elem_size,
      n);

  return total_bytes;
}

} // namespace

/*static*/ Result<TensorInfo> TensorInfo::create(
    Span<const int32_t> sizes,
    Span<const uint8_t> dim_order,
    executorch::aten::ScalarType scalar_type,
    const bool is_memory_planned,
    std::string_view name) {
  auto nbytes = calculate_nbytes(sizes, scalar_type);
  ET_CHECK_OR_RETURN_ERROR(
      nbytes.ok(),
      InvalidArgument,
      "Failed to calculate nbytes for TensorInfo");

  return TensorInfo(
      sizes, dim_order, scalar_type, is_memory_planned, name, nbytes.get());
}

TensorInfo::TensorInfo(
    Span<const int32_t> sizes,
    Span<const uint8_t> dim_order,
    executorch::aten::ScalarType scalar_type,
    const bool is_memory_planned,
    std::string_view name,
    size_t nbytes)
    : sizes_(sizes),
      dim_order_(dim_order),
      name_(name),
      scalar_type_(scalar_type),
      is_memory_planned_(is_memory_planned),
      nbytes_(nbytes) {}

Span<const int32_t> TensorInfo::sizes() const {
  return sizes_;
}

Span<const uint8_t> TensorInfo::dim_order() const {
  return dim_order_;
}

executorch::aten::ScalarType TensorInfo::scalar_type() const {
  return scalar_type_;
}

bool TensorInfo::is_memory_planned() const {
  return is_memory_planned_;
}

size_t TensorInfo::nbytes() const {
  return nbytes_;
}

std::string_view TensorInfo::name() const {
  return name_;
}

MethodMeta::MethodMeta(const executorch_flatbuffer::ExecutionPlan* s_plan)
    : s_plan_(s_plan) {}

const char* MethodMeta::name() const {
  return s_plan_->name()->c_str();
}

size_t MethodMeta::num_inputs() const {
  return s_plan_->inputs()->size();
}

Result<Tag> MethodMeta::input_tag(size_t index) const {
  auto num_inputs = this->num_inputs();
  ET_CHECK_OR_RETURN_ERROR(
      index < num_inputs,
      InvalidArgument,
      "index %zu out of range. num_inputs: %zu",
      index,
      num_inputs);
  auto input_index = s_plan_->inputs()->Get(index);
  size_t num_values = s_plan_->values()->size();
  ET_CHECK_OR_RETURN_ERROR(
      input_index >= 0 && static_cast<size_t>(input_index) < num_values,
      InvalidProgram,
      "internal value index %zd out of range [0,%zu) for input %zu",
      static_cast<ssize_t>(input_index),
      num_values,
      index);
  auto serialization_value = s_plan_->values()->Get(input_index);
  return get_tag(serialization_value, index);
}

Result<TensorInfo> MethodMeta::input_tensor_meta(size_t index) const {
  auto tag = this->input_tag(index);
  if (!tag.ok()) {
    return tag.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      tag.get() == Tag::Tensor,
      InvalidArgument,
      "Tag: %zu input: %zu is not Tensor",
      (size_t)tag.get(),
      index);
  auto input_index = s_plan_->inputs()->Get(index);
  // input_index was already validated by input_tag().
  auto tensor_value = s_plan_->values()->Get(input_index)->val_as_Tensor();
  return TensorInfo::create(
      Span<const int32_t>(
          tensor_value->sizes()->data(), tensor_value->sizes()->size()),
      Span<const uint8_t>(
          tensor_value->dim_order()->data(), tensor_value->dim_order()->size()),
      static_cast<executorch::aten::ScalarType>(tensor_value->scalar_type()),
      tensor_value->allocation_info() != nullptr ||
          tensor_value->data_buffer_idx() != 0 /* is_memory_planned */,
      std::string_view{nullptr, 0}); // Count constant returns as
                                     // memory planned.
}

size_t MethodMeta::num_outputs() const {
  return s_plan_->outputs()->size();
}

Result<Tag> MethodMeta::output_tag(size_t index) const {
  auto num_outputs = this->num_outputs();
  ET_CHECK_OR_RETURN_ERROR(
      index < num_outputs,
      InvalidArgument,
      "index %zu out of range. num_outputs: %zu",
      index,
      num_outputs);
  auto output_index = s_plan_->outputs()->Get(index);
  size_t num_values = s_plan_->values()->size();
  ET_CHECK_OR_RETURN_ERROR(
      output_index >= 0 && static_cast<size_t>(output_index) < num_values,
      InvalidProgram,
      "internal value index %zd out of range [0,%zu) for output %zu",
      static_cast<ssize_t>(output_index),
      num_values,
      index);
  auto serialization_value = s_plan_->values()->Get(output_index);
  return get_tag(serialization_value, index);
}

Result<TensorInfo> MethodMeta::output_tensor_meta(size_t index) const {
  auto tag = this->output_tag(index);
  if (!tag.ok()) {
    return tag.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      tag.get() == Tag::Tensor,
      InvalidArgument,
      "Tag: %zu output: %zu is not Tensor",
      (size_t)tag.get(),
      index);
  auto output_index = s_plan_->outputs()->Get(index);
  // output_index was already validated by output_tag().
  auto tensor_value = s_plan_->values()->Get(output_index)->val_as_Tensor();

  return TensorInfo::create(
      Span<const int32_t>(
          tensor_value->sizes()->data(), tensor_value->sizes()->size()),
      Span<const uint8_t>(
          tensor_value->dim_order()->data(), tensor_value->dim_order()->size()),
      static_cast<executorch::aten::ScalarType>(tensor_value->scalar_type()),
      tensor_value->allocation_info() != nullptr ||
          tensor_value->data_buffer_idx() != 0 /* is_memory_planned */,
      std::string_view{nullptr, 0}); // Count constant returns as
                                     // memory planned.
}

size_t MethodMeta::num_attributes() const {
  size_t counter = 0;
  auto values = s_plan_->values();
  for (size_t i = 0; i < values->size(); ++i) {
    auto value = values->Get(i);
    if (value->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
      auto tensor_value = value->val_as_Tensor();
      if (tensor_value->extra_tensor_info() != nullptr &&
          tensor_value->extra_tensor_info()->fully_qualified_name()->c_str() !=
              nullptr) {
        ++counter;
      }
    }
  }
  return counter;
}

Result<TensorInfo> MethodMeta::attribute_tensor_meta(size_t index) const {
  size_t counter = 0;
  auto values = s_plan_->values();
  for (size_t i = 0; i < values->size(); ++i) {
    auto value = values->Get(i);
    if (value->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
      auto tensor_value = value->val_as_Tensor();
      if (tensor_value->extra_tensor_info() != nullptr &&
          tensor_value->extra_tensor_info()->fully_qualified_name()->c_str() !=
              nullptr) {
        if (counter == index) {
          auto t_name =
              tensor_value->extra_tensor_info()->fully_qualified_name();
          // Count constant returns as memory planned
          return TensorInfo::create(
              Span<const int32_t>(
                  tensor_value->sizes()->data(), tensor_value->sizes()->size()),
              Span<const uint8_t>(
                  tensor_value->dim_order()->data(),
                  tensor_value->dim_order()->size()),
              static_cast<executorch::aten::ScalarType>(
                  tensor_value->scalar_type()),
              tensor_value->allocation_info() != nullptr ||
                  tensor_value->data_buffer_idx() != 0 /* is_memory_planned */,
              std::string_view{t_name->c_str(), t_name->size()});
        }
        ++counter;
      }
    }
  }
  ET_LOG(Error, "No attribute tensor found at index %zu", index);
  return Error::InvalidArgument;
}

size_t MethodMeta::num_memory_planned_buffers() const {
  if (s_plan_->non_const_buffer_sizes() == nullptr) {
    return 0;
  }
  const size_t size = s_plan_->non_const_buffer_sizes()->size();
  // Index zero is reserved internally, and we hide it from users. The actual
  // number of buffers is one fewer than the actual size of this list in the
  // program.
  return size > 0 ? size - 1 : 0;
}

Result<int64_t> MethodMeta::memory_planned_buffer_size(size_t index) const {
  auto num_buffers = this->num_memory_planned_buffers();
  ET_CHECK_OR_RETURN_ERROR(
      index < num_buffers,
      InvalidArgument,
      "index %zu out of range. num_buffers: %zu",
      index,
      num_buffers);
  // Index zero is reserved internally, and we hide it from users. Adjust the
  // provided index to point to one of the actual buffers.
  return s_plan_->non_const_buffer_sizes()->Get(index + 1);
}

bool MethodMeta::uses_backend(const char* backend_name) const {
  ET_CHECK_MSG(backend_name, "backend name is null");
  const auto delegates = s_plan_->delegates();
  for (size_t i = 0; i < delegates->size(); i++) {
    auto delegate = delegates->Get(i);
    auto backend_name_len = std::strlen(backend_name);
    auto delegate_id_len = delegate->id()->size();
    if (backend_name_len == delegate_id_len &&
        std::strncmp(delegate->id()->c_str(), backend_name, backend_name_len) ==
            0) {
      return true;
    }
  }
  return false;
}

size_t MethodMeta::num_backends() const {
  const auto delegates = s_plan_->delegates();
  return delegates ? delegates->size() : 0;
}

Result<const char*> MethodMeta::get_backend_name(size_t index) const {
  const auto count = num_backends();
  ET_CHECK_OR_RETURN_ERROR(
      index < count,
      InvalidArgument,
      "Index %zu out of range. num_backends: %zu",
      index,
      count);
  return s_plan_->delegates()->Get(index)->id()->c_str();
}

size_t MethodMeta::num_instructions() const {
  const auto chains = s_plan_->chains();
  if (chains == nullptr) {
    return 0;
  }
  const auto num_chains = chains->size();
  auto num_instructions = 0;
  for (size_t i = 0; i < num_chains; ++i) {
    auto s_chain = chains->Get(i);
    if (s_chain == nullptr) {
      continue;
    }
    auto s_instructions = s_chain->instructions();
    if (s_instructions != nullptr) {
      num_instructions += s_instructions->size();
    }
  }
  return num_instructions;
}
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
