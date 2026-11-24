/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/backends/arm/runtime/EthosUIOPlan.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::backends::arm::calculate_vela_io_bytes;
using executorch::backends::arm::copy_inputs;
using executorch::backends::arm::copy_outputs;
using executorch::backends::arm::IOPlacement;
using executorch::backends::arm::IOPlan;
using executorch::backends::arm::make_io_plan;
using executorch::backends::arm::parse_ethosu_arguments;
using executorch::backends::arm::VelaHandles;
using executorch::backends::arm::VelaIO;
using executorch::backends::arm::VelaIOs;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::etensor::TensorImpl;
using executorch::runtime::testing::TensorFactory;

namespace {

template <size_t N>
struct TestVelaIOs {
  int count;
  VelaIO io[N];
};

class EthosUIOPlanTest : public ::testing::Test {
 protected:
  // cppcheck-suppress unusedFunction
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
  }
};

class CharTensor {
 private:
  TensorImpl::SizesType sizes_[1];
  TensorImpl::DimOrderType dim_order_[1];
  TensorImpl::StridesType strides_[1];
  TensorImpl impl_;
  Tensor tensor_;

 public:
  CharTensor(char* data, int elements)
      : sizes_{elements},
        dim_order_{0},
        strides_{1},
        impl_(ScalarType::Char, 1, sizes_, data, dim_order_, strides_),
        tensor_(&impl_),
        value(tensor_) {}

  EValue value;
};

VelaHandles make_handles(VelaIOs* inputs, VelaIOs* outputs, bool cop2 = true) {
  static constexpr char kCop1Command[] = {'C', 'O', 'P', '1'};
  static constexpr char kCop2Command[] = {'C', 'O', 'P', '2'};
  VelaHandles handles{};
  handles.cmd_data = cop2 ? kCop2Command : kCop1Command;
  handles.cmd_data_size = sizeof(kCop2Command);
  handles.scratch_data_size = 64;
  handles.inputs = inputs;
  handles.outputs = outputs;
  return handles;
}

VelaIO make_io(int elements, int elem_size, int offset, int region) {
  return VelaIO{{1, 1, 1, 1, 1, elements}, elem_size, offset, region};
}

template <size_t N>
Span<uint8_t> scratch_span(char (&scratch)[N]) {
  return Span<uint8_t>(reinterpret_cast<uint8_t*>(scratch), N);
}

Result<IOPlan> make_resolved_io_plan(
    const VelaHandles& handles,
    Span<EValue*> args,
    Span<uint8_t> scratch,
    bool supports_direct_io = true) {
  auto arguments = parse_ethosu_arguments(handles, args);
  if (!arguments.ok()) {
    return arguments.error();
  }
  return make_io_plan(handles, *arguments, scratch, supports_direct_io);
}

void expect_copy_requirements(
    const IOPlan& io_plan,
    bool needs_input_copy,
    bool needs_output_copy) {
  EXPECT_EQ(io_plan.needs_input_copy, needs_input_copy);
  EXPECT_EQ(io_plan.needs_output_copy, needs_output_copy);
}

} // namespace

TEST_F(EthosUIOPlanTest, TreatsZeroPlaceholderDimensionsAsOne) {
  const VelaIO io{{0, 0, 0, 0, 0, 4}, 1, 0, 4};

  EXPECT_EQ(calculate_vela_io_bytes(io), 4);
}

TEST_F(EthosUIOPlanTest, PlansOutputWithZeroPlaceholderDimensions) {
  char output_data[4] = {};
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {{{0, 0, 0, 0, 0, 4}, 1, 0, 4}}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  EXPECT_FALSE(resolved->needs_output_copy);
  EXPECT_EQ(resolved->output_region_size, 4);
}

TEST_F(EthosUIOPlanTest, RejectsScratchAsTensorArgument) {
  char input_data[4] = {};
  char scratch_data[64] = {};
  char output_data[4] = {};
  CharTensor input(input_data, 4);
  CharTensor scratch(scratch_data, 64);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input.value, &scratch.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto io_plan = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch_data));

  ASSERT_FALSE(io_plan.ok());
  EXPECT_EQ(io_plan.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, ResolvesDirectSharedScratchIO) {
  char scratch[64] = {};
  CharTensor input(scratch, 4);
  CharTensor output(scratch + 4, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 4, 1)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, false);
  EXPECT_EQ(resolved->placement, IOPlacement::INSIDE_SCRATCH);
  EXPECT_EQ(resolved->input_region_base, scratch);
  EXPECT_EQ(resolved->input_region_size, 4);
  EXPECT_EQ(resolved->contiguous_output_base, scratch);
  EXPECT_EQ(resolved->output_region_size, 8);
}

TEST_F(EthosUIOPlanTest, UnsupportedPlatformDoesNotUsePlannedScratchForIO) {
  char scratch[64] = {};
  CharTensor input(scratch, 4);
  CharTensor output(scratch + 4, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 4, 1)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), Span<uint8_t>(), false);

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, true);
  EXPECT_EQ(resolved->input_region_base, nullptr);
  EXPECT_EQ(resolved->contiguous_output_base, nullptr);
}

TEST_F(EthosUIOPlanTest, RuntimeScratchRequiresInputCopy) {
  char scratch[64] = {};
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved->placement, IOPlacement::INSIDE_SCRATCH);
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_base, nullptr);
  EXPECT_EQ(resolved->input_region_size, 4);
}

TEST_F(EthosUIOPlanTest, ResolvesSharedScratchInputCopy) {
  char scratch[64] = {};
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  CharTensor output(scratch + 4, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 4, 1)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_base, nullptr);
  EXPECT_EQ(resolved->contiguous_output_base, scratch);
}

TEST_F(EthosUIOPlanTest, ResolvesSharedScratchOutputCopy) {
  char scratch[64] = {};
  char output_data[4] = {};
  CharTensor input(scratch, 4);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 4, 1)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, true);
  EXPECT_EQ(resolved->input_region_base, scratch);
  EXPECT_EQ(resolved->contiguous_output_base, nullptr);
}

TEST_F(EthosUIOPlanTest, ResolvesSharedScratchInputAndOutputCopies) {
  char scratch[64] = {};
  char input_data[4] = {};
  char output_data[4] = {};
  CharTensor input(input_data, 4);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 4, 1)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, true);
  EXPECT_EQ(resolved->placement, IOPlacement::INSIDE_SCRATCH);
}

TEST_F(EthosUIOPlanTest, ResolvesDirectSeparateRegions) {
  char input_data[4] = {};
  char output_data[4] = {};
  CharTensor input(input_data, 4);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, false);
  EXPECT_EQ(resolved->placement, IOPlacement::OUTSIDE_SCRATCH);
  EXPECT_EQ(resolved->input_region_base, input_data);
  EXPECT_EQ(resolved->contiguous_output_base, output_data);
}

TEST_F(EthosUIOPlanTest, ResolvesSeparateInputCopy) {
  char input_storage[12] = {};
  char output_data[4] = {};
  CharTensor input0(input_storage, 4);
  CharTensor input1(input_storage + 8, 4);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input0.value, &input1.value, &output.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 3), make_io(4, 1, 4, 3)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_size, 8);
  EXPECT_EQ(resolved->contiguous_output_base, output_data);
}

TEST_F(EthosUIOPlanTest, ResolvesSeparateOutputCopy) {
  char input_data[4] = {};
  char output_storage[12] = {};
  CharTensor input(input_data, 4);
  CharTensor output0(output_storage, 4);
  CharTensor output1(output_storage + 8, 4);
  EValue* args_data[] = {&input.value, &output0.value, &output1.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  TestVelaIOs<2> outputs{2, {make_io(4, 1, 0, 4), make_io(4, 1, 4, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, true);
  EXPECT_EQ(resolved->input_region_base, input_data);
  EXPECT_EQ(resolved->output_region_size, 8);
}

TEST_F(EthosUIOPlanTest, ResolvesSeparateInputAndOutputCopies) {
  char input_storage[12] = {};
  char output_storage[12] = {};
  CharTensor input0(input_storage, 4);
  CharTensor input1(input_storage + 8, 4);
  CharTensor output0(output_storage, 4);
  CharTensor output1(output_storage + 8, 4);
  EValue* args_data[] = {
      &input0.value, &input1.value, &output0.value, &output1.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 3), make_io(4, 1, 4, 3)}};
  TestVelaIOs<2> outputs{2, {make_io(4, 1, 0, 4), make_io(4, 1, 4, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, true);
  EXPECT_EQ(resolved->input_region_size, 8);
  EXPECT_EQ(resolved->output_region_size, 8);
}

TEST_F(EthosUIOPlanTest, SeparateRegionWithoutZeroOffsetRequiresCopy) {
  char input_storage[12] = {};
  CharTensor input0(input_storage + 4, 4);
  CharTensor input1(input_storage + 8, 4);
  EValue* args_data[] = {&input0.value, &input1.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 4, 3), make_io(4, 1, 8, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_size, 12);
}

TEST_F(EthosUIOPlanTest, RejectsNegativeIOOffset) {
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, -1, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsIOOffsetAndSizeOverflow) {
  // Positive factors of SIZE_MAX avoid overflowing the byte-count calculation.
  // An empty tensor avoids allocating storage for the synthetic Vela extent.
  CharTensor output(nullptr, 0);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{
      1,
      {VelaIO{
          {255,
           257,
           65537,
           1,
           sizeof(size_t) == 8 ? 641 : 1,
           sizeof(size_t) == 8 ? 6700417 : 1},
          1,
          1,
          4}}};
  ASSERT_EQ(
      calculate_vela_io_bytes(outputs.io[0]),
      std::numeric_limits<size_t>::max());
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, Cop1UsesSharedScratchAddressing) {
  char scratch[64] = {};
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved->placement, IOPlacement::INSIDE_SCRATCH);
  expect_copy_requirements(*resolved, true, false);
}

TEST_F(
    EthosUIOPlanTest,
    CopiesPartiallyMatchingSharedScratchInputsInEitherOrder) {
  for (bool direct_first : {false, true}) {
    char scratch[64] = {1, 2, 3, 4};
    char external[4] = {5, 6, 7, 8};
    CharTensor direct(scratch, 4);
    CharTensor staged(external, 4);
    EValue* args[] = {
        direct_first ? &direct.value : &staged.value,
        direct_first ? &staged.value : &direct.value};
    TestVelaIOs<2> inputs{
        2,
        {make_io(4, 1, direct_first ? 0 : 4, 1),
         make_io(4, 1, direct_first ? 4 : 0, 1)}};
    auto handles =
        make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
    auto plan = make_resolved_io_plan(
        handles, Span<EValue*>(args), scratch_span(scratch));
    ASSERT_TRUE(plan.ok());
    ASSERT_TRUE(plan->needs_input_copy);
    plan->input_region_base = scratch;

    EXPECT_EQ(copy_inputs(*plan), Error::Ok);
    EXPECT_EQ(scratch[0], 1);
    EXPECT_EQ(scratch[3], 4);
    EXPECT_EQ(std::memcmp(scratch + 4, external, sizeof(external)), 0);
  }
}

TEST_F(EthosUIOPlanTest, RejectsMissingSharedScratch) {
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidState);
}

TEST_F(EthosUIOPlanTest, RejectsInputTypeMismatch) {
  char scratch[64] = {};
  TensorFactory<ScalarType::Char> factory;
  auto input = factory.make({4}, {1, 2, 3, 4});
  EValue input_value(input);
  EValue* args_data[] = {&input_value};
  TestVelaIOs<1> inputs{1, {make_io(4, 2, 0, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, AcceptsSupportedIntegerInputTypes) {
  char scratch[64] = {};
  TensorFactory<ScalarType::Int> int_factory;
  TensorFactory<ScalarType::Byte> byte_factory;
  TensorFactory<ScalarType::Short> short_factory;
  TensorFactory<ScalarType::Bool> bool_factory;
  auto int_input = int_factory.make({2}, {1, 2});
  auto byte_input = byte_factory.make({3}, {3, 4, 5});
  auto short_input = short_factory.make({2}, {6, 7});
  auto bool_input = bool_factory.make({4}, {true, false, true, false});
  EValue int_value(int_input);
  EValue byte_value(byte_input);
  EValue short_value(short_input);
  EValue bool_value(bool_input);
  EValue* args_data[] = {&int_value, &byte_value, &short_value, &bool_value};
  TestVelaIOs<4> inputs{
      4,
      {make_io(2, 4, 0, 1),
       make_io(3, 1, 8, 1),
       make_io(2, 2, 11, 1),
       make_io(4, 1, 15, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_size, 19);
}

TEST_F(EthosUIOPlanTest, AcceptsMultidimensionalInput) {
  char scratch[64] = {};
  TensorFactory<ScalarType::Char> factory;
  auto input = factory.make({2, 3}, {1, 2, 3, 4, 5, 6});
  EValue input_value(input);
  EValue* args_data[] = {&input_value};
  TestVelaIOs<1> inputs{1, {VelaIO{{1, 1, 1, 1, 2, 3}, 1, 0, 1}}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, true, false);
  EXPECT_EQ(resolved->input_region_size, 6);
}

TEST_F(EthosUIOPlanTest, RejectsMultidimensionalElementCountMismatch) {
  char scratch[64] = {};
  TensorFactory<ScalarType::Char> factory;
  auto input = factory.make({2, 3}, {1, 2, 3, 4, 5, 6});
  EValue input_value(input);
  EValue* args_data[] = {&input_value};
  TestVelaIOs<1> inputs{1, {VelaIO{{1, 1, 1, 2, 2, 2}, 1, 0, 1}}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), scratch_span(scratch));

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, OutputSizeMismatchRequiresCopy) {
  char input_data[4] = {};
  char output_data[2] = {};
  CharTensor input(input_data, 4);
  CharTensor output(output_data, 2);
  EValue* args_data[] = {&input.value, &output.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, true);
  EXPECT_EQ(resolved->output_region_size, 4);
}

TEST_F(EthosUIOPlanTest, RejectsInvalidArgumentRange) {
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsNullOutputArgument) {
  char input_data[4] = {};
  char output_data[4] = {};
  CharTensor input(input_data, 4);
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&input.value, nullptr};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles = make_handles(
      reinterpret_cast<VelaIOs*>(&inputs),
      reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsInconsistentCop2Regions) {
  char input_data[8] = {};
  CharTensor input0(input_data, 4);
  CharTensor input1(input_data + 4, 4);
  EValue* args_data[] = {&input0.value, &input1.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 3), make_io(4, 1, 4, 1)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsUnsupportedCop2Region) {
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 2)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsNonTensorArgument) {
  EValue input_value;
  EValue* args_data[] = {&input_value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsNonEmptyTensorWithoutData) {
  CharTensor input(nullptr, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, AcceptsUint8Output) {
  TensorFactory<ScalarType::Byte> factory;
  auto output = factory.make({4}, {1, 2, 3, 4});
  EValue output_value(output);
  EValue* args_data[] = {&output_value};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_TRUE(resolved.ok());
  expect_copy_requirements(*resolved, false, false);
}

TEST_F(EthosUIOPlanTest, RejectsOutputElementSizeMismatch) {
  char output_data[4] = {};
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {make_io(2, 2, 0, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, RejectsZeroOutputElementSize) {
  char output_data[4] = {};
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {make_io(4, 0, 0, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto resolved =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());

  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.error(), Error::InvalidProgram);
}

TEST_F(EthosUIOPlanTest, CopiesInputsToVelaOffsets) {
  char input0_data[3] = {1, 2, 3};
  char input1_data[2] = {4, 5};
  char input_region[10] = {};
  CharTensor input0(input0_data, 3);
  CharTensor input1(input1_data, 2);
  EValue* args_data[] = {&input0.value, &input1.value};
  TestVelaIOs<2> inputs{2, {make_io(3, 1, 2, 3), make_io(2, 1, 8, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto io_plan =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());
  ASSERT_TRUE(io_plan.ok());
  io_plan->input_region_base = input_region;
  Error status = copy_inputs(*io_plan);

  ASSERT_EQ(status, Error::Ok);
  EXPECT_EQ(input_region[2], 1);
  EXPECT_EQ(input_region[3], 2);
  EXPECT_EQ(input_region[4], 3);
  EXPECT_EQ(input_region[8], 4);
  EXPECT_EQ(input_region[9], 5);
}

TEST_F(EthosUIOPlanTest, RejectsPartiallyOverlappingInputCopyBeforeWriting) {
  for (int source_offset : {0, 2}) {
    char scratch[64] = "ABCDEFGH";
    char before[sizeof(scratch)];
    std::memcpy(before, scratch, sizeof(scratch));
    CharTensor input(scratch + source_offset, 4);
    EValue* args[] = {&input.value};
    TestVelaIOs<1> inputs{1, {make_io(4, 1, 1, 1)}};
    auto handles =
        make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
    auto plan = make_resolved_io_plan(
        handles, Span<EValue*>(args), scratch_span(scratch));
    ASSERT_TRUE(plan.ok());
    plan->input_region_base = scratch;

    EXPECT_EQ(copy_inputs(*plan), Error::InvalidProgram);
    EXPECT_EQ(std::memcmp(scratch, before, sizeof(scratch)), 0);
  }
}

TEST_F(EthosUIOPlanTest, RejectsCopyDependenciesBeforeWritingAnyInput) {
  char scratch[64] = "AAAABBBB";
  char before[sizeof(scratch)];
  std::memcpy(before, scratch, sizeof(scratch));
  char external[4] = {'D', 'D', 'D', 'D'};
  CharTensor staged(external, 4);
  CharTensor first(scratch, 4);
  CharTensor second(scratch + 4, 4);
  EValue* args[] = {&staged.value, &first.value, &second.value};
  TestVelaIOs<3> inputs{
      3, {make_io(4, 1, 8, 1), make_io(4, 1, 4, 1), make_io(4, 1, 0, 1)}};
  auto handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
  auto plan = make_resolved_io_plan(
      handles, Span<EValue*>(args), scratch_span(scratch));
  ASSERT_TRUE(plan.ok());
  plan->input_region_base = scratch;

  EXPECT_EQ(copy_inputs(*plan), Error::InvalidProgram);
  EXPECT_EQ(std::memcmp(scratch, before, sizeof(scratch)), 0);
}

TEST_F(EthosUIOPlanTest, RejectsCopiesOverDirectInputsInEitherArgumentOrder) {
  for (bool direct_first : {false, true}) {
    char scratch[64] = "AAAA";
    char external[4] = {'B', 'B', 'B', 'B'};
    CharTensor direct(scratch, 4);
    CharTensor staged(external, 4);
    EValue* args[] = {
        direct_first ? &direct.value : &staged.value,
        direct_first ? &staged.value : &direct.value};
    TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 1), make_io(4, 1, 0, 1)}};
    auto handles =
        make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
    auto plan = make_resolved_io_plan(
        handles, Span<EValue*>(args), scratch_span(scratch));
    ASSERT_TRUE(plan.ok());
    plan->input_region_base = scratch;

    EXPECT_EQ(copy_inputs(*plan), Error::InvalidProgram);
    EXPECT_EQ(std::memcmp(scratch, "AAAA", 4), 0);
  }
}

TEST_F(EthosUIOPlanTest, RejectsDuplicateDestinationsWithDifferentSources) {
  char scratch[64] = {};
  char first_data[4] = {1, 2, 3, 4};
  char second_data[4] = {5, 6, 7, 8};
  CharTensor first(first_data, 4);
  CharTensor second(second_data, 4);
  EValue* args[] = {&first.value, &second.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 1), make_io(4, 1, 0, 1)}};
  auto handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
  auto plan = make_resolved_io_plan(
      handles, Span<EValue*>(args), scratch_span(scratch));
  ASSERT_TRUE(plan.ok());
  plan->input_region_base = scratch;

  EXPECT_EQ(copy_inputs(*plan), Error::InvalidProgram);
  EXPECT_EQ(scratch[0], 0);
  EXPECT_EQ(scratch[3], 0);
}

TEST_F(EthosUIOPlanTest, CopiesDuplicateDestinationsFromTheSameSource) {
  char scratch[64] = {};
  char source[4] = {1, 2, 3, 4};
  CharTensor input(source, 4);
  EValue* args[] = {&input.value, &input.value};
  TestVelaIOs<2> inputs{2, {make_io(4, 1, 0, 1), make_io(4, 1, 0, 1)}};
  auto handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr, false);
  auto plan = make_resolved_io_plan(
      handles, Span<EValue*>(args), scratch_span(scratch));
  ASSERT_TRUE(plan.ok());
  plan->input_region_base = scratch;

  EXPECT_EQ(copy_inputs(*plan), Error::Ok);
  EXPECT_EQ(std::memcmp(scratch, source, sizeof(source)), 0);
}

TEST_F(EthosUIOPlanTest, RejectsMissingInputCopyBuffer) {
  char input_data[4] = {};
  CharTensor input(input_data, 4);
  EValue* args_data[] = {&input.value};
  TestVelaIOs<1> inputs{1, {make_io(4, 1, 0, 3)}};
  VelaHandles handles =
      make_handles(reinterpret_cast<VelaIOs*>(&inputs), nullptr);

  auto io_plan = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), Span<uint8_t>(), false);
  ASSERT_TRUE(io_plan.ok());

  EXPECT_EQ(copy_inputs(*io_plan), Error::InvalidState);
}

TEST_F(EthosUIOPlanTest, CopiesOutputToEValue) {
  char output_region[8] = {0, 0, 1, 2, 3, 4, 0, 0};
  char output_data[4] = {};
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 2, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto io_plan =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());
  ASSERT_TRUE(io_plan.ok());
  io_plan->contiguous_output_base = output_region;
  Error status = copy_outputs(*io_plan);

  ASSERT_EQ(status, Error::Ok);
  EXPECT_EQ(output_data[0], 1);
  EXPECT_EQ(output_data[1], 2);
  EXPECT_EQ(output_data[2], 3);
  EXPECT_EQ(output_data[3], 4);
}

TEST_F(EthosUIOPlanTest, RejectsMissingOutputCopyBuffer) {
  char output_data[4] = {};
  CharTensor output(output_data, 4);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {make_io(4, 1, 0, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto io_plan = make_resolved_io_plan(
      handles, Span<EValue*>(args_data), Span<uint8_t>(), false);
  ASSERT_TRUE(io_plan.ok());

  EXPECT_EQ(copy_outputs(*io_plan), Error::InvalidState);
}

TEST_F(EthosUIOPlanTest, CopiesOnlyStagedPerOutputBuffers) {
  char direct_output[2] = {1, 2};
  char staged_output[2] = {3, 4};
  char copied_output[2] = {};
  CharTensor output0(direct_output, 2);
  CharTensor output1(copied_output, 2);
  EValue* args_data[] = {&output0.value, &output1.value};
  TestVelaIOs<2> outputs{2, {make_io(2, 1, 0, 4), make_io(2, 1, 0, 4)}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto io_plan =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());
  ASSERT_TRUE(io_plan.ok());
  char* output_data[] = {direct_output, staged_output};
  Error status = copy_outputs(*io_plan, output_data);

  ASSERT_EQ(status, Error::Ok);
  EXPECT_EQ(copied_output[0], 3);
  EXPECT_EQ(copied_output[1], 4);
}

TEST_F(EthosUIOPlanTest, RemovesVelaOutputPadding) {
  char output_region[8] = {1, 2, 3, 99, 4, 5, 6, 99};
  char output_data[6] = {};
  CharTensor output(output_data, 6);
  EValue* args_data[] = {&output.value};
  TestVelaIOs<1> outputs{1, {{{1, 1, 1, 1, 2, 4}, 1, 0, 4}}};
  VelaHandles handles =
      make_handles(nullptr, reinterpret_cast<VelaIOs*>(&outputs));

  auto io_plan =
      make_resolved_io_plan(handles, Span<EValue*>(args_data), Span<uint8_t>());
  ASSERT_TRUE(io_plan.ok());
  io_plan->contiguous_output_base = output_region;
  Error status = copy_outputs(*io_plan);

  ASSERT_EQ(status, Error::Ok);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(output_data[i], i + 1);
  }
}
