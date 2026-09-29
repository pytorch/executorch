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
  // Empty means each dim is static at its size.
  std::vector<int64_t> lower_bounds;
};

flatbuffers::Offset<native_backend::Method> make_state_method(
    flatbuffers::FlatBufferBuilder& builder,
    const std::string& method_name,
    const StateSpec& state) {
  std::vector<flatbuffers::Offset<native_backend::Dim>> dimensions;
  dimensions.reserve(state.sizes.size());
  for (size_t i = 0; i < state.sizes.size(); ++i) {
    const int64_t lower =
        state.lower_bounds.empty() ? state.sizes[i] : state.lower_bounds[i];
    dimensions.push_back(
        native_backend::CreateDim(builder, lower, state.sizes[i]));
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

// forward(x) = test.op(x, weight), with weight a bound parameter.
Method make_op_method() {
  Method method;
  method.name = "forward";
  method.graph.values = {
      Value("weight", kFloat, {2}),
      Value("x", kFloat, {2}),
      Value("y", kFloat, {2})};
  method.graph.values[0].role = ValueRole::Parameter;
  method.graph.values[1].role = ValueRole::UserInput;
  method.graph.nodes = {
      Node{
          .name = "weight",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 0}}},
      Node{
          .name = "x",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 1}}},
      Node{
          .name = "y",
          .target = "test.op",
          .inputs =
              {{.name = "self", .arg = TensorArg{1}},
               {.name = "other", .arg = TensorArg{0}}},
          .outputs = {{.value_id = 2}}},
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.arg = TensorArg{2}}}},
  };
  method.graph.input_ids = {1};
  method.graph.output_ids = {2};
  method.graph.initialize_schedule();
  method.graph.rebuild_def_use();
  method.output_specs = {OutputSpec{}};
  method.data_bindings.push_back(DataBinding{
      /*value_id=*/0,
      ValueRole::Parameter,
      "weight",
      /*has_data=*/true,
      /*mutated=*/false});
  return method;
}

TEST(ValidationTest, ValidateMethodConstants_OpMethod_Succeeds) {
  EXPECT_NO_THROW(validate_method_constants(make_op_method(), make_package()));
}

TEST(ValidationTest, ValidateMethodConstants_StructureRejects) {
  Method stale_def_use = make_op_method();
  stale_def_use.graph.values[2].consumer_ids.push_back(2);
  Method int_list_refs = make_op_method();
  int_list_refs.graph.nodes[2].inputs.push_back(
      {.name = "size", .arg = IntListArg{{2, 1}, {kInvalid}}});
  Method writes_parameter = make_op_method();
  writes_parameter.graph.nodes[2].inputs[1].mutated = true;
  Method aliases_non_input = make_op_method();
  aliases_non_input.graph.nodes[2].inputs.pop_back();
  aliases_non_input.graph.rebuild_def_use();
  aliases_non_input.graph.values[2].alias_id = 0;
  Method duplicate_key = make_op_method();
  duplicate_key.data_bindings.push_back(duplicate_key.data_bindings[0]);
  const std::vector<std::pair<std::string, Method>> cases = {
      {"stale def-use", std::move(stale_def_use)},
      {"int list refs", std::move(int_list_refs)},
      {"parameter written in place", std::move(writes_parameter)},
      {"alias of a non-input", std::move(aliases_non_input)},
      {"duplicate binding key", std::move(duplicate_key)},
  };
  const Package package = make_package();
  for (const auto& [name, method] : cases) {
    SCOPED_TRACE(name);
    EXPECT_THROW(
        validate_method_constants(method, package), std::runtime_error);
  }
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

TEST(ValidationTest, ValidateProgramState_DifferentLowerBounds_Throws) {
  StateSpec dynamic;
  dynamic.lower_bounds = {1};
  const Program program = make_program(StateSpec{}, dynamic);

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

Package make_q4_package() {
  auto tensors = testing::make_safetensors(
      R"({"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},)"
      R"("scale":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
      /*payload_size=*/12);
  return Package::load(OwnedBytes::from_vector(testing::make_zip({
      {kProgramEntry, {'N', 'P', 'T', 'G'}},
      {kSafeTensorsEntry, std::move(tensors)},
  })));
}

Method make_q4_method(
    QuantScheme quant,
    std::vector<int64_t> sizes = {2, 4},
    std::string key = "weight") {
  TensorMeta meta{ScalarType::Byte, std::move(sizes)};
  meta.quant = std::move(quant);
  Method method = make_method();
  method.graph.values.clear();
  method.graph.values.emplace_back("weight", std::move(meta));
  method.data_bindings[0].key = std::move(key);
  return method;
}

AffineGroupQuant q4_scheme() {
  return AffineGroupQuant{
      .scale_data_key = "scale",
      .quant_min = -8,
      .quant_max = 7,
      .group_size = 4};
}

TEST(ValidationTest, ValidateMethodConstants_AffineGroup_Succeeds) {
  EXPECT_NO_THROW(validate_method_constants(
      make_q4_method(q4_scheme()), make_q4_package()));
}

TEST(ValidationTest, ValidateMethodConstants_PerChannelAffineGroup_Succeeds) {
  AffineGroupQuant per_channel = q4_scheme();
  per_channel.group_size = 0;
  EXPECT_NO_THROW(validate_method_constants(
      make_q4_method(per_channel), make_q4_package()));
}

TEST(ValidationTest, ValidateMethodConstants_AffineGroupRejects) {
  const Package package = make_q4_package();
  AffineGroupQuant indivisible = q4_scheme();
  indivisible.group_size = 3;
  AffineGroupQuant non_power_of_two = q4_scheme();
  non_power_of_two.quant_max = 6;
  AffineGroupQuant scale_count = q4_scheme();
  scale_count.group_size = 2;
  AffineGroupQuant missing_zero_point = q4_scheme();
  missing_zero_point.zero_point_data_key = "missing";
  AffineGroupQuant no_scale = q4_scheme();
  no_scale.scale_data_key.clear();
  const std::vector<std::pair<std::string, Method>> cases = {
      {"indivisible group", make_q4_method(indivisible)},
      {"non-power-of-two range", make_q4_method(non_power_of_two)},
      {"scale count", make_q4_method(scale_count)},
      {"missing zero point", make_q4_method(missing_zero_point)},
      {"no scale", make_q4_method(no_scale)},
      {"packed byte count", make_q4_method(q4_scheme(), {2, 8})},
      {"non-Byte constant", make_q4_method(q4_scheme(), {2, 4}, "scale")},
      {"empty codec", make_q4_method(PackedQuant{})},
  };
  for (const auto& [name, method] : cases) {
    SCOPED_TRACE(name);
    EXPECT_THROW(
        validate_method_constants(method, package), std::runtime_error);
  }
}

} // namespace
} // namespace ptn
