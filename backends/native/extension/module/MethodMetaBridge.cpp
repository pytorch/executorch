// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/MethodMetaBridge.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/schema/program_generated.h>

namespace executorch::extension::native_module {
namespace {

executorch_flatbuffer::ScalarType to_flatbuffer_scalar_type(
    ptn::ScalarType type) {
  switch (type) {
    case ptn::ScalarType::Byte:
      return executorch_flatbuffer::ScalarType::BYTE;
    case ptn::ScalarType::Char:
      return executorch_flatbuffer::ScalarType::CHAR;
    case ptn::ScalarType::Short:
      return executorch_flatbuffer::ScalarType::SHORT;
    case ptn::ScalarType::Int:
      return executorch_flatbuffer::ScalarType::INT;
    case ptn::ScalarType::Long:
      return executorch_flatbuffer::ScalarType::LONG;
    case ptn::ScalarType::Half:
      return executorch_flatbuffer::ScalarType::HALF;
    case ptn::ScalarType::Float:
      return executorch_flatbuffer::ScalarType::FLOAT;
    case ptn::ScalarType::Double:
      return executorch_flatbuffer::ScalarType::DOUBLE;
    case ptn::ScalarType::Bool:
      return executorch_flatbuffer::ScalarType::BOOL;
    case ptn::ScalarType::BFloat16:
      return executorch_flatbuffer::ScalarType::BFLOAT16;
    case ptn::ScalarType::UInt16:
      return executorch_flatbuffer::ScalarType::UINT16;
    case ptn::ScalarType::UInt32:
      return executorch_flatbuffer::ScalarType::UINT32;
    case ptn::ScalarType::UInt64:
      return executorch_flatbuffer::ScalarType::UINT64;
  }
  throw std::runtime_error("unrecognized PTN scalar type");
}

flatbuffers::Offset<executorch_flatbuffer::EValue> make_tensor_value(
    flatbuffers::FlatBufferBuilder& builder,
    const ptn::TensorInfo& info) {
  std::vector<int32_t> serialized_sizes;
  serialized_sizes.reserve(info.sizes().size());
  for (const int64_t size : info.sizes()) {
    if (size > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(
          "PTN tensor size is not representable in ET metadata");
    }
    serialized_sizes.push_back(static_cast<int32_t>(size));
  }
  const auto sizes = builder.CreateVector(serialized_sizes);
  const auto dim_order =
      builder.CreateVector(info.dim_order().data(), info.dim_order().size());
  const auto tensor = executorch_flatbuffer::CreateTensor(
      builder,
      to_flatbuffer_scalar_type(info.dtype()),
      /*storage_offset=*/0,
      sizes,
      dim_order);
  return executorch_flatbuffer::CreateEValue(
      builder, executorch_flatbuffer::KernelTypes::Tensor, tensor.Union());
}

ET_RUNTIME_NAMESPACE::MethodMeta validated_view(
    const std::vector<uint8_t>& bytes) {
  auto result =
      ET_RUNTIME_NAMESPACE::MethodMeta::from_serialized_execution_plan(
          bytes.data(), bytes.size());
  if (!result.ok()) {
    throw std::runtime_error("native metadata bridge produced an invalid plan");
  }
  return *result;
}

} // namespace

std::unique_ptr<MethodMetaBridge> MethodMetaBridge::create(
    const ptn::MethodMeta& meta) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>> values;
  std::vector<int32_t> input_indices;
  std::vector<int32_t> output_indices;
  values.reserve(meta.inputs().size() + meta.outputs().size());
  input_indices.reserve(meta.inputs().size());
  output_indices.reserve(meta.outputs().size());

  for (const ptn::TensorInfo& input : meta.inputs()) {
    input_indices.push_back(static_cast<int32_t>(values.size()));
    values.push_back(make_tensor_value(builder, input));
  }
  for (const ptn::TensorInfo& output : meta.outputs()) {
    output_indices.push_back(static_cast<int32_t>(values.size()));
    values.push_back(make_tensor_value(builder, output));
  }

  const auto name = builder.CreateString(meta.name());
  const auto serialized_values = builder.CreateVector(values);
  const auto inputs = builder.CreateVector(input_indices);
  const auto outputs = builder.CreateVector(output_indices);
  const auto chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  const auto operators = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  const auto delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  const auto buffer_sizes = builder.CreateVector(std::vector<int64_t>{0});
  const auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      name,
      /*container_meta_type=*/0,
      serialized_values,
      inputs,
      outputs,
      chains,
      operators,
      delegates,
      buffer_sizes);
  builder.Finish(plan);

  std::vector<uint8_t> bytes(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
  return std::unique_ptr<MethodMetaBridge>(
      new MethodMetaBridge(std::move(bytes)));
}

MethodMetaBridge::MethodMetaBridge(std::vector<uint8_t> bytes)
    : bytes_(std::move(bytes)), view_(validated_view(bytes_)) {}

} // namespace executorch::extension::native_module
