// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Validation.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>
#include <flatbuffers/flatbuffers.h>

namespace ptn {
namespace {

Package make_package(
    const std::string& header =
        R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})") {
  auto tensors = testing::make_safetensors(
      header,
      /*payload_size=*/8);
  return Package::load(OwnedBytes::from_vector(testing::make_zip({
      {kProgramEntry, {'N', 'P', 'T', 'G'}},
      {kSafeTensorsEntry, std::move(tensors)},
  })));
}

Method make_method(
    ScalarType dtype = kFloat,
    std::vector<int64_t> sizes = {2},
    std::vector<int32_t> dim_order = {}) {
  Method method;
  method.name = "forward";
  method.graph.values.emplace_back(
      "weight", TensorMeta{dtype, std::move(sizes), std::move(dim_order)});
  method.data_bindings.push_back(DataBinding{
      /*value_id=*/0,
      ValueRole::Parameter,
      "weight",
      /*has_data=*/true,
      /*mutated=*/false});
  return method;
}

struct StateSpec {
  native_backend::InputKind kind = native_backend::InputKind::PARAMETER;
  bool has_data = true;
  bool mutated = false;
  native_backend::ScalarType dtype = native_backend::ScalarType::FLOAT;
  std::vector<int64_t> sizes{2};
  std::vector<int32_t> dim_order;
};

flatbuffers::Offset<native_backend::Method> make_state_method(
    flatbuffers::FlatBufferBuilder& builder,
    const std::string& method_name,
    const StateSpec& state) {
  std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
  dimensions.reserve(state.sizes.size());
  for (const int64_t size : state.sizes) {
    dimensions.push_back(native_backend::CreateDim(builder, size, size));
  }
  const auto value_name = builder.CreateString("state");
  const auto metadata = native_backend::CreateTensorMeta(
      builder,
      state.dtype,
      builder.CreateVector(dimensions),
      builder.CreateVector(state.dim_order));
  const auto tensor =
      native_backend::CreateTensorValue(builder, value_name, metadata);
  const auto graph = native_backend::CreateGraph(
      builder,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::Node>>{}),
      /*inputs=*/0,
      /*outputs=*/0,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::TensorValue>>{
              tensor}));

  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<native_backend::NamedTensorRef>>>
      constants = 0;
  flatbuffers::Offset<flatbuffers::Vector<
      flatbuffers::Offset<native_backend::MutableBufferSpec>>>
      mutable_buffers = 0;
  if (state.has_data) {
    const auto binding = native_backend::CreateNamedTensorRef(
        builder,
        value_name,
        builder.CreateString("shared.state"),
        metadata,
        state.kind,
        state.mutated);
    constants = builder.CreateVector(
        std::vector<flatbuffers::Offset<native_backend::NamedTensorRef>>{
            binding});
  } else {
    mutable_buffers = builder.CreateVector(
        std::vector<flatbuffers::Offset<native_backend::MutableBufferSpec>>{
            native_backend::CreateMutableBufferSpec(
                builder, value_name, builder.CreateString("shared.state"))});
  }
  return native_backend::CreateMethod(
      builder,
      builder.CreateString(method_name),
      graph,
      constants,
      /*output_specs=*/0,
      mutable_buffers);
}

Program make_program(const StateSpec& first, const StateSpec& second) {
  flatbuffers::FlatBufferBuilder builder;
  const auto methods = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Method>>{
          make_state_method(builder, "first", first),
          make_state_method(builder, "second", second)});
  const auto program = native_backend::CreateProgram(
      builder, builder.CreateString("1.0"), methods);
  native_backend::FinishProgramBuffer(builder, program);
  return Program::load(builder.GetBufferPointer(), builder.GetSize());
}

TEST(ValidationTest, ValidateMethodConstants_MatchingBinding_Succeeds) {
  const Package package = make_package();
  const Method method = make_method();

  EXPECT_NO_THROW(validate_method_constants(method, package));
}

TEST(ValidationTest, ValidateMethodConstants_MissingBinding_Throws) {
  const Package package = make_package();
  Method method = make_method();
  method.data_bindings[0].key = "missing";

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

TEST(ValidationTest, ValidateMethodConstants_DtypeMismatch_Throws) {
  const Package package = make_package();
  const Method method = make_method(kInt);

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

TEST(ValidationTest, ValidateMethodConstants_NonContiguousBinding_Throws) {
  const Package package = make_package(
      R"({"weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}})");
  const Method contiguous = make_method(kFloat, {1, 2}, {0, 1});
  const Method noncontiguous = make_method(kFloat, {1, 2}, {1, 0});

  EXPECT_NO_THROW(validate_method_constants(contiguous, package));
  EXPECT_THROW(
      validate_method_constants(noncontiguous, package), std::runtime_error);
}

TEST(ValidationTest, ValidateMethodConstants_InvalidDtype_Throws) {
  const Package package = make_package();
  Method method;
  method.name = "forward";
  method.graph.values.emplace_back(
      "input", TensorMeta{static_cast<ScalarType>(127), {2}, {}});

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

TEST(ValidationTest, ValidateMethodConstants_InvalidInputId_Throws) {
  const Package package = make_package();
  Method method = make_method();
  method.graph.input_ids.push_back(7);

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

TEST(ValidationTest, ValidateMethodConstants_InvalidDimOrder_Throws) {
  const Package package = make_package();
  Method method;
  method.name = "forward";
  method.graph.values.emplace_back("input", TensorMeta{kFloat, {1, 2}, {0, 0}});

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

TEST(ValidationTest, ValidateProgramState_MatchingDefinitions_Succeeds) {
  const StateSpec state;
  const Program program = make_program(state, state);

  EXPECT_NO_THROW(validate_program_state(program));
}

TEST(ValidationTest, ValidateProgramState_EquivalentLayouts_Succeeds) {
  StateSpec explicit_identity;
  explicit_identity.sizes = {1, 2};
  explicit_identity.dim_order = {0, 1};
  StateSpec implicit_identity = explicit_identity;
  implicit_identity.dim_order.clear();
  const Program program = make_program(implicit_identity, explicit_identity);

  EXPECT_NO_THROW(validate_program_state(program));
}

TEST(ValidationTest, ValidateProgramState_DifferentRoles_Throws) {
  StateSpec constant;
  constant.kind = native_backend::InputKind::CONSTANT_TENSOR;
  const Program program = make_program(StateSpec{}, constant);

  EXPECT_THROW(validate_program_state(program), std::runtime_error);
}

TEST(ValidationTest, ValidateProgramState_DataBackedAndZeroInitialized_Throws) {
  StateSpec persistent_buffer;
  persistent_buffer.kind = native_backend::InputKind::BUFFER;
  persistent_buffer.mutated = true;
  StateSpec zero_initialized = persistent_buffer;
  zero_initialized.has_data = false;
  const Program program = make_program(persistent_buffer, zero_initialized);

  EXPECT_THROW(validate_program_state(program), std::runtime_error);
}

TEST(ValidationTest, ValidateProgramState_DifferentMutationUse_Succeeds) {
  StateSpec immutable_buffer;
  immutable_buffer.kind = native_backend::InputKind::BUFFER;
  StateSpec mutable_buffer = immutable_buffer;
  mutable_buffer.mutated = true;
  const Program program = make_program(immutable_buffer, mutable_buffer);

  EXPECT_NO_THROW(validate_program_state(program));
}

TEST(ValidationTest, ValidateProgramState_DifferentDtypes_Throws) {
  StateSpec integer;
  integer.dtype = native_backend::ScalarType::INT;
  const Program program = make_program(StateSpec{}, integer);

  EXPECT_THROW(validate_program_state(program), std::runtime_error);
}

TEST(ValidationTest, ValidateProgramState_DifferentShapes_Throws) {
  StateSpec other_shape;
  other_shape.sizes = {3};
  const Program program = make_program(StateSpec{}, other_shape);

  EXPECT_THROW(validate_program_state(program), std::runtime_error);
}

TEST(ValidationTest, ValidateProgramState_DifferentLayouts_Throws) {
  StateSpec other_layout;
  other_layout.sizes = {1, 2};
  other_layout.dim_order = {1, 0};
  StateSpec identity = other_layout;
  identity.dim_order = {0, 1};
  const Program program = make_program(identity, other_layout);

  EXPECT_THROW(validate_program_state(program), std::runtime_error);
}

} // namespace
} // namespace ptn
