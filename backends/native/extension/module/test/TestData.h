// cppcheck-suppress-file useStlAlgorithm

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>
#include <flatbuffers/flatbuffers.h>

namespace executorch::extension::native_module::testing {

// cppcheck-suppress-begin useStlAlgorithm
inline std::vector<uint8_t> make_tensor_package(
    const std::string& method_name = "forward",
    const std::vector<int64_t>& sizes = {2}) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
  dimensions.reserve(sizes.size());
  for (const int64_t size : sizes) {
    dimensions.push_back(native_backend::CreateDim(builder, size, size));
  }

  const auto serialized_sizes = builder.CreateVector(dimensions);
  const auto input_meta = native_backend::CreateTensorMeta(
      builder, native_backend::ScalarType::FLOAT, serialized_sizes);
  const auto output_meta = native_backend::CreateTensorMeta(
      builder, native_backend::ScalarType::FLOAT, serialized_sizes);
  const auto input = native_backend::CreateTensorValue(
      builder, builder.CreateString("input"), input_meta);
  const auto output = native_backend::CreateTensorValue(
      builder, builder.CreateString("output"), output_meta);
  const auto tensor_values = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::TensorValue>>{
          input, output});
  const auto nodes = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Node>>{});
  const auto inputs = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{
          builder.CreateString("input")});
  const auto outputs = builder.CreateVector(
      std::vector<flatbuffers::Offset<flatbuffers::String>>{
          builder.CreateString("output")});
  const auto graph = native_backend::CreateGraph(
      builder, nodes, inputs, outputs, tensor_values);
  const auto output_spec = native_backend::CreateOutputSpec(
      builder,
      builder.CreateString("output"),
      native_backend::OutputKind::USER_OUTPUT,
      builder.CreateString(""));
  const auto output_specs = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::OutputSpec>>{
          output_spec});
  const auto method = native_backend::CreateMethod(
      builder, builder.CreateString(method_name), graph, 0, output_specs);
  const auto methods = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Method>>{method});
  const auto program = native_backend::CreateProgram(
      builder, builder.CreateString("1.0"), methods);
  native_backend::FinishProgramBuffer(builder, program);

  std::vector<uint8_t> program_bytes(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
  return ptn::testing::make_zip(
      {{ptn::kProgramEntry, std::move(program_bytes)}});
}

inline std::vector<uint8_t> make_tensor_package_with_bad_constant_checksum() {
  flatbuffers::FlatBufferBuilder builder;
  const auto graph = native_backend::CreateGraph(
      builder,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::Node>>{}),
      /*inputs=*/0,
      /*outputs=*/0,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::TensorValue>>{}));
  const auto method = native_backend::CreateMethod(
      builder, builder.CreateString("forward"), graph);
  const auto program = native_backend::CreateProgram(
      builder,
      builder.CreateString("1.0"),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::Method>>{method}));
  native_backend::FinishProgramBuffer(builder, program);
  std::vector<uint8_t> program_bytes(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
  std::vector<uint8_t> constants = ptn::testing::make_safetensors(
      R"({"unused":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      /*payload_size=*/4);

  const size_t second_local =
      30 + std::string(ptn::kProgramEntry).size() + program_bytes.size();
  const size_t central = second_local + 30 +
      std::string(ptn::kSafeTensorsEntry).size() + constants.size();
  const size_t second_central =
      central + 46 + std::string(ptn::kProgramEntry).size();
  std::vector<uint8_t> bytes = ptn::testing::make_zip({
      {ptn::kProgramEntry, std::move(program_bytes)},
      {ptn::kSafeTensorsEntry, std::move(constants)},
  });
  ptn::testing::write_le<uint32_t>(bytes, second_local + 14, /*value=*/0);
  ptn::testing::write_le<uint32_t>(bytes, second_central + 16, /*value=*/0);
  return bytes;
}
// cppcheck-suppress-end useStlAlgorithm

} // namespace executorch::extension::native_module::testing
