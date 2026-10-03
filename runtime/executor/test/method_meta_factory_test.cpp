/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/method_meta.h>

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/schema/program_generated.h>

namespace executorch::runtime {
namespace {

static_assert(sizeof(MethodMeta) == sizeof(void*));

TEST(MethodMetaFactoryTest, FromSerializedExecutionPlan_PreservesView) {
  flatbuffers::FlatBufferBuilder builder;
  const auto name = builder.CreateString("native_forward");
  const auto values = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{});
  const auto inputs = builder.CreateVector(std::vector<int32_t>{});
  const auto outputs = builder.CreateVector(std::vector<int32_t>{});
  const auto delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  const auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      name,
      /*container_meta_type=*/0,
      values,
      inputs,
      outputs,
      /*chains=*/0,
      /*operators=*/0,
      delegates);
  builder.Finish(plan);
  const auto meta = MethodMeta::from_serialized_execution_plan(
      builder.GetBufferPointer(), builder.GetSize());

  ASSERT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "native_forward");
  EXPECT_EQ(meta->num_inputs(), 0);
  EXPECT_EQ(meta->num_outputs(), 0);
  EXPECT_EQ(meta->num_attributes(), 0);
  EXPECT_EQ(meta->num_memory_planned_buffers(), 0);
  EXPECT_EQ(meta->num_backends(), 0);
  EXPECT_EQ(meta->num_instructions(), 0);
}

TEST(MethodMetaFactoryTest, FromSerializedExecutionPlan_RejectsTruncation) {
  flatbuffers::FlatBufferBuilder builder;
  const auto plan = executorch_flatbuffer::CreateExecutionPlan(builder);
  builder.Finish(plan);

  const auto meta = MethodMeta::from_serialized_execution_plan(
      builder.GetBufferPointer(), builder.GetSize() - 1);

  EXPECT_EQ(meta.error(), Error::InvalidProgram);
}

TEST(MethodMetaFactoryTest, FromSerializedExecutionPlan_RejectsNull) {
  const auto meta = MethodMeta::from_serialized_execution_plan(
      /*data=*/nullptr, /*size=*/0);

  EXPECT_EQ(meta.error(), Error::InvalidProgram);
}

TEST(
    MethodMetaFactoryTest,
    FromSerializedExecutionPlan_RejectsMissingRequiredMetadata) {
  flatbuffers::FlatBufferBuilder builder;
  const auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder, builder.CreateString("native_forward"));
  builder.Finish(plan);

  const auto meta = MethodMeta::from_serialized_execution_plan(
      builder.GetBufferPointer(), builder.GetSize());

  EXPECT_EQ(meta.error(), Error::InvalidProgram);
}

TEST(MethodMetaFactoryTest, FromSerializedExecutionPlan_RejectsInvalidInput) {
  flatbuffers::FlatBufferBuilder builder;
  const auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      builder.CreateString("native_forward"),
      /*container_meta_type=*/0,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{}),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{}),
      /*chains=*/0,
      /*operators=*/0,
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{}));
  builder.Finish(plan);

  const auto meta = MethodMeta::from_serialized_execution_plan(
      builder.GetBufferPointer(), builder.GetSize());

  EXPECT_EQ(meta.error(), Error::InvalidProgram);
}

} // namespace
} // namespace executorch::runtime
