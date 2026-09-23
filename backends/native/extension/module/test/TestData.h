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
    const std::vector<std::string>& method_names,
    size_t num_inputs = 1,
    const std::vector<int64_t>& sizes = {2},
    bool bind_missing_constant = false,
    const std::string& version = "1.0",
    const std::vector<int32_t>& dim_order = {}) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<native_backend::Method>> serialized_methods;
  serialized_methods.reserve(method_names.size());
  for (const std::string& method_name : method_names) {
    std::vector<flatbuffers::Offset<native_backend::TensorValue>> tensor_values;
    std::vector<flatbuffers::Offset<flatbuffers::String>> input_names;
    tensor_values.reserve(num_inputs + 1);
    input_names.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
      dimensions.reserve(sizes.size());
      for (const int64_t size : sizes) {
        dimensions.push_back(native_backend::CreateDim(builder, size, size));
      }
      const auto input_name = builder.CreateString("input" + std::to_string(i));
      input_names.push_back(input_name);
      tensor_values.push_back(native_backend::CreateTensorValue(
          builder,
          input_name,
          native_backend::CreateTensorMeta(
              builder,
              native_backend::ScalarType::FLOAT,
              builder.CreateVector(dimensions),
              builder.CreateVector(dim_order))));
    }
    std::vector<flatbuffers::Offset<native_backend::Dim>> output_dimensions;
    output_dimensions.reserve(sizes.size());
    for (const int64_t size : sizes) {
      output_dimensions.push_back(
          native_backend::CreateDim(builder, size, size));
    }
    const auto output_name = builder.CreateString("output");
    tensor_values.push_back(native_backend::CreateTensorValue(
        builder,
        output_name,
        native_backend::CreateTensorMeta(
            builder,
            native_backend::ScalarType::FLOAT,
            builder.CreateVector(output_dimensions),
            builder.CreateVector(dim_order))));
    if (bind_missing_constant) {
      std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
      dimensions.reserve(sizes.size());
      for (const int64_t size : sizes) {
        dimensions.push_back(native_backend::CreateDim(builder, size, size));
      }
      tensor_values.push_back(native_backend::CreateTensorValue(
          builder,
          builder.CreateString("weight"),
          native_backend::CreateTensorMeta(
              builder,
              native_backend::ScalarType::FLOAT,
              builder.CreateVector(dimensions))));
    }
    const auto graph = native_backend::CreateGraph(
        builder,
        builder.CreateVector(
            std::vector<flatbuffers::Offset<native_backend::Node>>{}),
        builder.CreateVector(input_names),
        builder.CreateVector(
            std::vector<flatbuffers::Offset<flatbuffers::String>>{output_name}),
        builder.CreateVector(tensor_values));
    const auto output_spec = native_backend::CreateOutputSpec(
        builder,
        output_name,
        native_backend::OutputKind::USER_OUTPUT,
        builder.CreateString(""));
    flatbuffers::Offset<flatbuffers::Vector<
        flatbuffers::Offset<native_backend::NamedTensorRef>>>
        constants = 0;
    if (bind_missing_constant) {
      std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
      dimensions.reserve(sizes.size());
      for (const int64_t size : sizes) {
        dimensions.push_back(native_backend::CreateDim(builder, size, size));
      }
      const auto binding = native_backend::CreateNamedTensorRef(
          builder,
          builder.CreateString("weight"),
          builder.CreateString("missing.weight"),
          native_backend::CreateTensorMeta(
              builder,
              native_backend::ScalarType::FLOAT,
              builder.CreateVector(dimensions)),
          native_backend::InputKind::PARAMETER);
      constants = builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::NamedTensorRef>>{
              binding});
    }
    serialized_methods.push_back(native_backend::CreateMethod(
        builder,
        builder.CreateString(method_name),
        graph,
        constants,
        builder.CreateVector(
            std::vector<flatbuffers::Offset<native_backend::OutputSpec>>{
                output_spec})));
  }
  const auto methods = builder.CreateVector(serialized_methods);
  const auto program = native_backend::CreateProgram(
      builder, builder.CreateString(version), methods);
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

inline std::vector<uint8_t> make_missing_constant_package() {
  return make_tensor_package(
      {"forward"}, /*num_inputs=*/1, {2}, /*bind_missing_constant=*/true);
}

inline std::vector<uint8_t> make_tensor_package_with_version(
    const std::string& version) {
  return make_tensor_package(
      {"forward"},
      /*num_inputs=*/1,
      {2},
      /*bind_missing_constant=*/false,
      version);
}

inline std::vector<uint8_t> make_tensor_package(
    const std::string& method_name = "forward",
    const std::vector<int64_t>& sizes = {2}) {
  return make_tensor_package({method_name}, /*num_inputs=*/1, sizes);
}

} // namespace executorch::extension::native_module::testing
