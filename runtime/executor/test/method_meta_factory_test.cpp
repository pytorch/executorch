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

TEST(MethodMetaFactoryTest, FromValidatedExecutionPlan_PreservesView) {
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
  const auto* serialized =
      flatbuffers::GetRoot<executorch_flatbuffer::ExecutionPlan>(
          builder.GetBufferPointer());

  const MethodMeta meta =
      MethodMeta::from_validated_execution_plan(*serialized);

  EXPECT_STREQ(meta.name(), "native_forward");
  EXPECT_EQ(meta.num_inputs(), 0);
  EXPECT_EQ(meta.num_outputs(), 0);
  EXPECT_EQ(meta.num_attributes(), 0);
  EXPECT_EQ(meta.num_memory_planned_buffers(), 0);
  EXPECT_EQ(meta.num_backends(), 0);
  EXPECT_EQ(meta.num_instructions(), 0);
}

} // namespace
} // namespace executorch::runtime
