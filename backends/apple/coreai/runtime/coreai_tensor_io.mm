/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_tensor_io.h"

#import "coreai_error.h"

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace executorch::backends::coreai {
namespace {
using namespace executorch::runtime;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;

struct TensorMetadata final {
  ETCoreAIScalarType scalar_type;
  size_t bytes;
  size_t rank;
};

struct ValidatedArguments final {
  std::vector<TensorMetadata> tensors;
};

struct OutputPlan final {
  std::vector<SizesType> shape;
  size_t bytes = 0;
  void* destination = nullptr;
};

Result<ETCoreAIScalarType> scalar_type(ScalarType type) {
  switch (type) {
    case ScalarType::Half:
      return ETCoreAIScalarTypeFloat16;
    case ScalarType::Float:
      return ETCoreAIScalarTypeFloat32;
    default:
      ET_LOG(Error, "Core AI runtime supports FP16 and FP32 tensors only");
      return Error::NotSupported;
  }
}

Result<size_t> bridge_scalar_width(ETCoreAIScalarType type) {
  switch (type) {
    case ETCoreAIScalarTypeFloat16:
      return size_t(2);
    case ETCoreAIScalarTypeFloat32:
      return size_t(4);
    default:
      return Error::Internal;
  }
}

Result<TensorMetadata> validate_tensor(const Tensor& tensor) {
  auto type = scalar_type(tensor.scalar_type());
  if (!type.ok()) {
    return type.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      tensor.device_type() == executorch::aten::DeviceType::CPU,
      NotSupported,
      "Core AI requires CPU-accessible tensor storage");
  ET_CHECK_OR_RETURN_ERROR(
      tensor.dim() >= 0 &&
          static_cast<size_t>(tensor.dim()) <= kTensorDimensionLimit,
      InvalidArgument,
      "Invalid Core AI tensor rank");
  ET_CHECK_OR_RETURN_ERROR(
      tensor_is_default_dim_order(tensor),
      NotSupported,
      "Core AI requires default tensor dimension order");
  size_t elements = 1;
  size_t expected_stride = 1;
  for (size_t i = tensor.sizes().size(); i > 0; --i) {
    const auto size = tensor.sizes()[i - 1];
    ET_CHECK_OR_RETURN_ERROR(
        size >= 0, InvalidArgument, "Negative Core AI tensor dimension");
    ET_CHECK_OR_RETURN_ERROR(
        tensor.strides()[i - 1] >= 0 &&
            static_cast<size_t>(tensor.strides()[i - 1]) == expected_stride,
        NotSupported,
        "Core AI requires contiguous tensor storage");
    ET_CHECK_OR_RETURN_ERROR(
        !c10::mul_overflows(elements, static_cast<size_t>(size), &elements) &&
            !c10::mul_overflows(
                expected_stride,
                std::max(static_cast<size_t>(size), size_t(1)),
                &expected_stride),
        InvalidArgument,
        "Core AI tensor size overflow");
  }
  size_t bytes = 0;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::mul_overflows(
          elements, static_cast<size_t>(tensor.element_size()), &bytes) &&
          bytes <= static_cast<size_t>(std::numeric_limits<NSInteger>::max()) &&
          bytes == tensor.nbytes(),
      InvalidArgument,
      "Invalid Core AI tensor byte count");
  return TensorMetadata{type.get(), bytes, tensor.sizes().size()};
}

Result<ValidatedArguments> validate_arguments(
    Span<EValue*> args,
    size_t input_count,
    size_t output_count) {
  ET_CHECK_OR_RETURN_ERROR(
      args.size() >= input_count && args.size() - input_count == output_count,
      InvalidArgument,
      "Core AI argument count mismatch");
  ValidatedArguments validated;
  validated.tensors.reserve(args.size());
  for (EValue* arg : args) {
    ET_CHECK_OR_RETURN_ERROR(
        arg && arg->isTensor(),
        InvalidArgument,
        "Core AI arguments must be tensors");
    auto metadata = validate_tensor(arg->toTensor());
    if (!metadata.ok()) {
      return metadata.error();
    }
    validated.tensors.push_back(metadata.get());
  }
  return validated;
}

Result<NSArray<ETCoreAIInputTensor*>*> borrow_inputs(
    Span<EValue*> args,
    const ValidatedArguments& validated,
    size_t input_count) {
  NSMutableArray<ETCoreAIInputTensor*>* inputs =
      [NSMutableArray arrayWithCapacity:input_count];
  for (size_t i = 0; i < input_count; ++i) {
    const auto& tensor = args[i]->toTensor();
    const auto& metadata = validated.tensors[i];
    ET_CHECK_OR_RETURN_ERROR(
        metadata.bytes == 0 || tensor.const_data_ptr() != nullptr,
        InvalidArgument,
        "Core AI input has no storage");
    NSMutableArray<NSNumber*>* shape =
        [NSMutableArray arrayWithCapacity:tensor.dim()];
    for (auto size : tensor.sizes()) {
      [shape addObject:@(size)];
    }
    [inputs addObject:[[ETCoreAIInputTensor alloc]
                          initWithBytes:tensor.const_data_ptr()
                              byteCount:metadata.bytes
                                  shape:shape
                             scalarType:metadata.scalar_type]];
  }
  return inputs;
}

Result<NSArray<ETCoreAITensor*>*> wait_for_outputs(
    id<ETCoreAISession> session,
    NSArray<ETCoreAIInputTensor*>* inputs) {
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block NSArray<ETCoreAITensor*>* outputs = nil;
  __block NSError* error = nil;
  [session executeInputs:inputs
              completion:^(NSArray<ETCoreAITensor*>* result, NSError* failure) {
                outputs = result;
                error = failure;
                dispatch_semaphore_signal(ready);
              }];
  // Returning earlier would let the SDK outlive borrowed input storage.
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  NSArray<ETCoreAITensor*>* completed = outputs;
  NSError* failure = error;
  // The SDK may retain the completion block after invoking it.
  outputs = nil;
  error = nil;
  if (failure != nil) {
    return execution_error(failure);
  }
  return completed;
}

Result<std::vector<OutputPlan>> validate_bridge_outputs(
    NSArray<ETCoreAITensor*>* outputs,
    const ValidatedArguments& validated,
    size_t input_count,
    size_t output_count) {
  ET_CHECK_OR_RETURN_ERROR(
      outputs != nil && outputs.count == output_count,
      Internal,
      "Core AI output count mismatch");
  std::vector<OutputPlan> plans(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    ETCoreAITensor* output = outputs[i];
    ET_CHECK_OR_RETURN_ERROR(
        [output isKindOfClass:ETCoreAITensor.class],
        Internal,
        "Invalid Core AI output tensor");
    auto width = bridge_scalar_width(output.scalarType);
    if (!width.ok()) {
      return width.error();
    }
    const auto& expected = validated.tensors[input_count + i];
    ET_CHECK_OR_RETURN_ERROR(
        output.scalarType == expected.scalar_type &&
            output.shape.count == expected.rank,
        InvalidArgument,
        "Core AI output type or rank mismatch");
    OutputPlan& plan = plans[i];
    plan.bytes = width.get();
    plan.shape.reserve(output.shape.count);
    for (id size in output.shape) {
      ET_CHECK_OR_RETURN_ERROR(
          [size isKindOfClass:NSNumber.class] && [size longLongValue] >= 0 &&
              [size longLongValue] <= std::numeric_limits<SizesType>::max() &&
              [size isEqualToNumber:@([size longLongValue])],
          Internal,
          "Invalid Core AI output dimension");
      const auto dimension = static_cast<SizesType>([size longLongValue]);
      plan.shape.push_back(dimension);
      ET_CHECK_OR_RETURN_ERROR(
          !c10::mul_overflows(
              plan.bytes, static_cast<size_t>(dimension), &plan.bytes),
          Internal,
          "Core AI output size overflow");
    }
    ET_CHECK_OR_RETURN_ERROR(
        plan.bytes == output.byteCount,
        Internal,
        "Core AI output byte count mismatch");
  }
  return plans;
}

Error resize_outputs(
    std::vector<OutputPlan>& plans,
    Span<EValue*> args,
    size_t input_count) {
  for (size_t i = 0; i < plans.size(); ++i) {
    auto& tensor = args[input_count + i]->toTensor();
    auto& shape = plans[i].shape;
    const Error resized =
        resize_tensor(tensor, ArrayRef<SizesType>(shape.data(), shape.size()));
    if (resized != Error::Ok) {
      return resized == Error::NotSupported ? Error::InvalidArgument : resized;
    }
  }
  return Error::Ok;
}

Error validate_resized_outputs(
    std::vector<OutputPlan>& plans,
    Span<EValue*> args,
    const ValidatedArguments& validated,
    size_t input_count) {
  for (size_t i = 0; i < plans.size(); ++i) {
    auto& tensor = args[input_count + i]->toTensor();
    auto& plan = plans[i];
    ET_CHECK_OR_RETURN_ERROR(
        tensor.dim() >= 0 &&
            static_cast<size_t>(tensor.dim()) == plan.shape.size() &&
            std::equal(
                plan.shape.begin(), plan.shape.end(), tensor.sizes().begin()),
        InvalidArgument,
        "Core AI output shape changed during resize");
    auto metadata = validate_tensor(tensor);
    if (!metadata.ok()) {
      return metadata.error();
    }
    ET_CHECK_OR_RETURN_ERROR(
        metadata->scalar_type ==
                validated.tensors[input_count + i].scalar_type &&
            metadata->bytes == plan.bytes &&
            (plan.bytes == 0 || tensor.mutable_data_ptr() != nullptr),
        InvalidArgument,
        "Core AI output has insufficient storage");
    plan.destination = tensor.mutable_data_ptr();
  }
  return Error::Ok;
}

bool storage_overlaps(
    const void* left,
    size_t left_size,
    const void* right,
    size_t right_size) {
  if (left_size == 0 || right_size == 0) {
    return false;
  }
  const auto a = reinterpret_cast<uintptr_t>(left);
  const auto b = reinterpret_cast<uintptr_t>(right);
  return a <= b ? b - a < left_size : a - b < right_size;
}

bool outputs_need_staging(
    const std::vector<OutputPlan>& plans,
    NSArray<ETCoreAIInputTensor*>* inputs) {
  for (size_t i = 0; i < plans.size(); ++i) {
    const auto& plan = plans[i];
    for (ETCoreAIInputTensor* input in inputs) {
      if (storage_overlaps(
              plan.destination, plan.bytes, input.bytes, input.byteCount)) {
        return true;
      }
    }
    for (size_t j = 0; j < i; ++j) {
      if (storage_overlaps(
              plan.destination,
              plan.bytes,
              plans[j].destination,
              plans[j].bytes)) {
        return true;
      }
    }
  }
  return false;
}

Error copy_direct(
    NSArray<ETCoreAITensor*>* outputs,
    const std::vector<OutputPlan>& plans) {
  for (size_t i = 0; i < plans.size(); ++i) {
    [outputs[i] copyBytesTo:plans[i].destination];
  }
  return Error::Ok;
}

Error copy_staged(
    NSArray<ETCoreAITensor*>* outputs,
    const std::vector<OutputPlan>& plans) {
  size_t total_bytes = 0;
  for (const auto& plan : plans) {
    ET_CHECK_OR_RETURN_ERROR(
        plan.bytes <= std::numeric_limits<size_t>::max() - total_bytes,
        MemoryAllocationFailed,
        "Core AI output staging size overflow");
    total_bytes += plan.bytes;
  }
  std::unique_ptr<uint8_t[]> staging(new (std::nothrow) uint8_t[total_bytes]);
  ET_CHECK_OR_RETURN_ERROR(
      staging != nullptr,
      MemoryAllocationFailed,
      "Cannot stage Core AI outputs");
  size_t offset = 0;
  for (size_t i = 0; i < plans.size(); ++i) {
    [outputs[i] copyBytesTo:staging.get() + offset];
    offset += plans[i].bytes;
  }
  offset = 0;
  for (const auto& plan : plans) {
    if (plan.bytes > 0) {
      memcpy(plan.destination, staging.get() + offset, plan.bytes);
    }
    offset += plan.bytes;
  }
  return Error::Ok;
}

} // namespace

Error execute_session(
    id<ETCoreAISession> session,
    Span<EValue*> args,
    size_t input_count,
    size_t output_count) {
  ET_CHECK_OR_RETURN_ERROR(
      session != nil, DelegateInvalidHandle, "Missing Core AI session");
  auto validated = validate_arguments(args, input_count, output_count);
  if (!validated.ok()) {
    return validated.error();
  }
  auto inputs = borrow_inputs(args, validated.get(), input_count);
  if (!inputs.ok()) {
    return inputs.error();
  }
  auto outputs = wait_for_outputs(session, inputs.get());
  if (!outputs.ok()) {
    return outputs.error();
  }
  auto plans = validate_bridge_outputs(
      outputs.get(), validated.get(), input_count, output_count);
  if (!plans.ok()) {
    return plans.error();
  }
  ET_CHECK_OK_OR_RETURN_ERROR(resize_outputs(plans.get(), args, input_count));
  // Finish every resize before copying so a late failure cannot write results.
  ET_CHECK_OK_OR_RETURN_ERROR(validate_resized_outputs(
      plans.get(), args, validated.get(), input_count));
  return outputs_need_staging(plans.get(), inputs.get())
      ? copy_staged(outputs.get(), plans.get())
      : copy_direct(outputs.get(), plans.get());
}

} // namespace executorch::backends::coreai
