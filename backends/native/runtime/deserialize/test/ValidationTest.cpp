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
      "weight",
      TensorMeta{
          dtype, std::move(sizes), std::move(dim_order), {}, std::nullopt});
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

Node make_node(
    std::string name,
    OpKind op_kind,
    std::vector<NamedArgument> inputs,
    std::vector<ValueId> outputs,
    std::string target = "") {
  Node node;
  node.name = std::move(name);
  node.op_kind = op_kind;
  node.target = std::move(target);
  node.inputs = std::move(inputs);
  for (const ValueId id : outputs) {
    Output output;
    output.value_id = id;
    node.outputs.push_back(std::move(output));
  }
  return node;
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
      make_node("weight", OpKind::Placeholder, {}, {0}),
      make_node("x", OpKind::Placeholder, {}, {1}),
      make_node(
          "y",
          OpKind::CallFunction,
          {{"self", TensorArg{1}}, {"other", TensorArg{0}}},
          {2},
          "test.op"),
      make_node("output", OpKind::Output, {{"", TensorArg{2}}}, {}),
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
      "input",
      TensorMeta{static_cast<ScalarType>(127), {2}, {}, {}, std::nullopt});

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
  method.graph.values.emplace_back(
      "input", TensorMeta{kFloat, {1, 2}, {0, 0}, {}, std::nullopt});

  EXPECT_THROW(validate_method_constants(method, package), std::runtime_error);
}

constexpr char kQuantizedHeader[] =
    R"({"weight":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},)"
    R"("weight_scales":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]},)"
    R"("weight_zeros":{"dtype":"I32","shape":[2,2],"data_offsets":[24,40]}})";

Package make_quantized_package() {
  auto tensors = testing::make_safetensors(
      kQuantizedHeader,
      /*payload_size=*/40);
  return Package::load(OwnedBytes::from_vector(testing::make_zip({
      {kProgramEntry, {'N', 'P', 'T', 'G'}},
      {kSafeTensorsEntry, std::move(tensors)},
  })));
}

// An int4 [2, 8] weight in groups of 4, with its scale and zero-point tensors
// bound as ordinary constants of the method.
AffineQuant weight_quant() {
  return AffineQuant{
      "weight_scales",
      kFloat,
      /*quant_min=*/-8,
      /*quant_max=*/7,
      /*group_size=*/4,
      "weight_zeros",
      kInt,
      /*axis=*/-1,
      AffineGranularity::PerGroup};
}

// An int4 [2, 8] weight in groups of 4, with its scale and zero-point tensors
// bound as ordinary constants of the method.
Method make_quantized_method(AffineQuant quant = weight_quant()) {
  Method method;
  method.name = "forward";
  TensorMeta weight{kByte, {2, 8}, {}, {}, std::nullopt};
  weight.quant = std::move(quant);
  method.graph.values.emplace_back("weight", std::move(weight));
  method.graph.values.emplace_back(
      "weight_scales", TensorMeta{kFloat, {2, 2}, {}, {}, std::nullopt});
  method.graph.values.emplace_back(
      "weight_zeros", TensorMeta{kInt, {2, 2}, {}, {}, std::nullopt});
  for (ValueId id = 0; id < 3; ++id) {
    method.data_bindings.push_back(DataBinding{
        id,
        ValueRole::Parameter,
        method.graph.values[id].name,
        /*has_data=*/true,
        /*mutated=*/false});
  }
  return method;
}

TEST(ValidationTest, ValidateMethodConstants_QuantizedWeight_Succeeds) {
  const Package package = make_quantized_package();
  const Method method = make_quantized_method();

  EXPECT_NO_THROW(validate_method_constants(method, package));
  ASSERT_NE(find_data_binding(method, "weight_scales"), nullptr);
  EXPECT_EQ(find_data_binding(method, "weight_scales")->value_id, 1);
  EXPECT_EQ(find_data_binding(method, "missing"), nullptr);
}

TEST(ValidationTest, ValidateMethodConstants_QuantParameterBindingRejects) {
  Method unbound_scale = make_quantized_method();
  unbound_scale.data_bindings.erase(unbound_scale.data_bindings.begin() + 1);
  Method unbound_zero_point = make_quantized_method();
  unbound_zero_point.data_bindings.pop_back();
  AffineQuant half_scale = weight_quant();
  half_scale.scale_dtype = kHalf;
  Method mutable_scale = make_quantized_method();
  mutable_scale.data_bindings[1].role = ValueRole::Buffer;
  mutable_scale.data_bindings[1].mutated = true;
  const std::vector<std::pair<std::string, Method>> cases = {
      {"unbound scale", std::move(unbound_scale)},
      {"unbound zero point", std::move(unbound_zero_point)},
      {"scale dtype", make_quantized_method(half_scale)},
      {"mutable scale", std::move(mutable_scale)},
  };
  const Package package = make_quantized_package();
  for (const auto& [name, method] : cases) {
    SCOPED_TRACE(name);
    EXPECT_THROW(
        validate_method_constants(method, package), std::runtime_error);
  }
}

Method with_quantized_input(AffineQuant quant) {
  Method method = make_quantized_method();
  TensorMeta input{kByte, {2, 8}, {}, {}, std::nullopt};
  input.quant = std::move(quant);
  method.graph.values.emplace_back("x", std::move(input));
  method.graph.values.back().role = ValueRole::UserInput;
  return method;
}

// Quantized graph values (e.g. user I/O) also need bound parameters, though
// only constants have their packed size checked against the package.
TEST(ValidationTest, ValidateMethodConstants_QuantizedGraphValue) {
  const Package package = make_quantized_package();
  AffineQuant missing_scale = weight_quant();
  missing_scale.scale_data_key = "missing";
  const Method bound = with_quantized_input(weight_quant());
  const Method unbound = with_quantized_input(missing_scale);

  EXPECT_NO_THROW(validate_method_constants(bound, package));
  EXPECT_THROW(validate_method_constants(unbound, package), std::runtime_error);
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
  TensorMeta meta{ScalarType::Byte, std::move(sizes), {}, {}, std::nullopt};
  meta.quant = std::move(quant);
  Method method = make_method();
  method.graph.values.clear();
  method.graph.values.emplace_back("weight", std::move(meta));
  method.graph.values.emplace_back(
      "scale", TensorMeta{kFloat, {2}, {}, {}, std::nullopt});
  method.data_bindings[0].key = std::move(key);
  method.data_bindings.push_back(DataBinding{
      /*value_id=*/1,
      ValueRole::Parameter,
      "scale",
      /*has_data=*/true,
      /*mutated=*/false});
  return method;
}

AffineQuant q4_scheme() {
  return AffineQuant{
      .scale_data_key = "scale",
      .scale_dtype = kFloat,
      .quant_min = -8,
      .quant_max = 7,
      .group_size = 4,
      .zero_point_data_key = "",
      .zero_point_dtype = kInt,
      .axis = -1,
      .granularity = AffineGranularity::PerGroup};
}

TEST(ValidationTest, ValidateMethodConstants_AffineQuant_Succeeds) {
  EXPECT_NO_THROW(validate_method_constants(
      make_q4_method(q4_scheme()), make_q4_package()));
}

TEST(ValidationTest, ValidateMethodConstants_PerAxisAffineQuant_Succeeds) {
  AffineQuant per_channel = q4_scheme();
  per_channel.group_size = 0;
  per_channel.axis = 0;
  per_channel.granularity = AffineGranularity::PerAxis;
  EXPECT_NO_THROW(validate_method_constants(
      make_q4_method(per_channel), make_q4_package()));
}

TEST(ValidationTest, ValidateMethodConstants_PositiveLastGroupAxis_Succeeds) {
  AffineQuant group = q4_scheme();
  group.axis = 1;

  EXPECT_NO_THROW(validate_method_constants(
      make_q4_method(group), make_q4_package()));
}

TEST(ValidationTest, ValidateMethodConstants_AffineQuantRejects) {
  const Package package = make_q4_package();
  AffineQuant indivisible = q4_scheme();
  indivisible.group_size = 3;
  AffineQuant non_power_of_two = q4_scheme();
  non_power_of_two.quant_max = 6;
  AffineQuant scale_count = q4_scheme();
  scale_count.group_size = 2;
  AffineQuant missing_zero_point = q4_scheme();
  missing_zero_point.zero_point_data_key = "missing";
  AffineQuant no_scale = q4_scheme();
  no_scale.scale_data_key.clear();
  AffineQuant group_on_first_axis = q4_scheme();
  group_on_first_axis.axis = 0;
  AffineQuant group_without_size = q4_scheme();
  group_without_size.group_size = 0;
  AffineQuant axis_with_size = q4_scheme();
  axis_with_size.granularity = AffineGranularity::PerAxis;
  AffineQuant axis_out_of_range = q4_scheme();
  axis_out_of_range.granularity = AffineGranularity::PerAxis;
  axis_out_of_range.group_size = 0;
  axis_out_of_range.axis = -3;
  AffineQuant tensor_with_size = q4_scheme();
  tensor_with_size.granularity = AffineGranularity::PerTensor;
  const std::vector<std::pair<std::string, Method>> cases = {
      {"indivisible group", make_q4_method(indivisible)},
      {"non-power-of-two range", make_q4_method(non_power_of_two)},
      {"scale count", make_q4_method(scale_count)},
      {"missing zero point", make_q4_method(missing_zero_point)},
      {"no scale", make_q4_method(no_scale)},
      {"group on first axis", make_q4_method(group_on_first_axis)},
      {"group without size", make_q4_method(group_without_size)},
      {"per-axis with group size", make_q4_method(axis_with_size)},
      {"per-axis out of range", make_q4_method(axis_out_of_range)},
      {"per-tensor with group size", make_q4_method(tensor_with_size)},
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

Package make_dense_package() {
  auto tensors = testing::make_safetensors(
      R"({"weight":{"dtype":"I8","shape":[2,1,2,2],"data_offsets":[0,8]},)"
      R"("scale":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},)"
      R"("act_scale":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}})",
      /*payload_size=*/20);
  return Package::load(OwnedBytes::from_vector(testing::make_zip({
      {kProgramEntry, {'N', 'P', 'T', 'G'}},
      {kSafeTensorsEntry, std::move(tensors)},
  })));
}

AffineQuant int8_per_channel() {
  return AffineQuant{
      .scale_data_key = "scale",
      .scale_dtype = kFloat,
      .quant_min = -127,
      .quant_max = 127,
      .group_size = 0,
      .zero_point_data_key = "",
      .zero_point_dtype = kInt,
      .axis = 0,
      .granularity = AffineGranularity::PerAxis};
}

AffineQuant int16_per_tensor() {
  return AffineQuant{
      .scale_data_key = "act_scale",
      .scale_dtype = kFloat,
      .quant_min = -32768,
      .quant_max = 32767,
      .group_size = 0,
      .zero_point_data_key = "",
      .zero_point_dtype = kInt,
      .axis = 0,
      .granularity = AffineGranularity::PerTensor};
}

// A dense int8 conv weight quantized per output channel and an unbound int16
// activation quantized per tensor, each scaled by a bound constant.
Method make_dense_method(
    AffineQuant weight_quant = int8_per_channel(),
    AffineQuant activation_quant = int16_per_tensor(),
    ScalarType activation_dtype = kShort) {
  TensorMeta weight{kChar, {2, 1, 2, 2}, {}, {}, std::nullopt};
  weight.quant = std::move(weight_quant);
  TensorMeta activation{activation_dtype, {1, 4}, {}, {}, std::nullopt};
  activation.quant = std::move(activation_quant);
  Method method;
  method.name = "forward";
  method.graph.values.emplace_back("weight", std::move(weight));
  method.graph.values.emplace_back(
      "scale", TensorMeta{kFloat, {2}, {}, {}, std::nullopt});
  method.graph.values.emplace_back(
      "act_scale", TensorMeta{kFloat, {1}, {}, {}, std::nullopt});
  method.graph.values.emplace_back("activation", std::move(activation));
  for (ValueId id = 0; id < 3; ++id) {
    method.data_bindings.push_back(DataBinding{
        id,
        ValueRole::Parameter,
        method.graph.values[id].name,
        /*has_data=*/true,
        /*mutated=*/false});
  }
  return method;
}

TEST(ValidationTest, ValidateMethodConstants_DenseAffine_Succeeds) {
  EXPECT_NO_THROW(
      validate_method_constants(make_dense_method(), make_dense_package()));
}

TEST(ValidationTest, ValidateMethodConstants_NegativePerAxis_Succeeds) {
  for (const int32_t axis : {-4, -1}) {
    SCOPED_TRACE(axis);
    AffineQuant per_axis = int8_per_channel();
    per_axis.axis = axis;
    EXPECT_NO_THROW(validate_method_constants(
        make_dense_method(per_axis), make_dense_package()));
  }
}

TEST(ValidationTest, ValidateMethodConstants_NonLeadingPerAxis_Succeeds) {
  AffineQuant per_axis = int8_per_channel();
  per_axis.axis = 2;

  EXPECT_NO_THROW(validate_method_constants(
      make_dense_method(per_axis), make_dense_package()));
}

TEST(ValidationTest, ValidateMethodConstants_TransposedWeightPerAxis_Succeeds) {
  auto tensors = testing::make_safetensors(
      R"({"weight":{"dtype":"I8","shape":[4,2],"data_offsets":[0,8]},)"
      R"("scale":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},)"
      R"("act_scale":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}})",
      /*payload_size=*/20);
  const Package package = Package::load(OwnedBytes::from_vector(testing::make_zip({
      {kProgramEntry, {'N', 'P', 'T', 'G'}},
      {kSafeTensorsEntry, std::move(tensors)},
  })));
  AffineQuant per_axis = int8_per_channel();
  per_axis.axis = 1;
  Method method = make_dense_method(per_axis);
  method.graph.values[0].tensor_meta().sizes = {4, 2};

  EXPECT_NO_THROW(validate_method_constants(method, package));
}

TEST(ValidationTest, ValidateMethodConstants_DenseAffineRejects) {
  const Package package = make_dense_package();
  AffineQuant exceeds_int8 = int8_per_channel();
  exceeds_int8.quant_max = 128;
  AffineQuant axis_out_of_range = int8_per_channel();
  axis_out_of_range.axis = 4;
  AffineQuant negative_axis_out_of_range = int8_per_channel();
  negative_axis_out_of_range.axis = -5;
  AffineQuant group_with_axis = int8_per_channel();
  group_with_axis.group_size = 2;
  group_with_axis.axis = 1;
  group_with_axis.granularity = AffineGranularity::PerGroup;
  AffineQuant channel_scale_count = int8_per_channel();
  channel_scale_count.axis = 1;
  AffineQuant per_tensor_scale_count = int16_per_tensor();
  per_tensor_scale_count.scale_data_key = "scale";
  AffineQuant unbound_scale = int16_per_tensor();
  unbound_scale.scale_data_key = "missing";
  AffineQuant per_tensor_with_axis = int16_per_tensor();
  per_tensor_with_axis.axis = -1;
  const std::vector<std::pair<std::string, Method>> cases = {
      {"range exceeds int8", make_dense_method(exceeds_int8)},
      {"axis out of range", make_dense_method(axis_out_of_range)},
      {"negative axis out of range",
       make_dense_method(negative_axis_out_of_range)},
      {"group on non-last axis", make_dense_method(group_with_axis)},
      {"per-channel scale count", make_dense_method(channel_scale_count)},
      {"per-tensor scale count",
       make_dense_method(int8_per_channel(), per_tensor_scale_count)},
      {"unbound activation scale",
       make_dense_method(int8_per_channel(), unbound_scale)},
      {"per-tensor with axis",
       make_dense_method(int8_per_channel(), per_tensor_with_axis)},
      {"non-integer storage",
       make_dense_method(int8_per_channel(), int16_per_tensor(), kFloat)},
  };
  for (const auto& [name, method] : cases) {
    SCOPED_TRACE(name);
    EXPECT_THROW(
        validate_method_constants(method, package), std::runtime_error);
  }
}

} // namespace
} // namespace ptn
