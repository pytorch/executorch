/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/method_meta.h>

#include <cstdlib>
#include <limits>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
using torch::executor::util::FileDataLoader;

namespace executorch {
namespace runtime {
namespace testing {
// Provides access to private TensorInfo methods.
class TensorInfoTestFriend final {
 public:
  ET_NODISCARD static TensorInfo get(
      Span<const int32_t> sizes,
      Span<const uint8_t> dim_order,
      executorch::aten::ScalarType scalar_type,
      const bool is_memory_planned,
      executorch::aten::string_view name) {
    return TensorInfo::create(
               Span<const int32_t>(sizes.data(), sizes.size()),
               Span<const uint8_t>(dim_order.data(), dim_order.size()),
               scalar_type,
               is_memory_planned,
               name)
        .get();
  }
};
} // namespace testing
} // namespace runtime
} // namespace executorch

class MethodMetaTest : public ::testing::Test {
 protected:
  void load_program(const char* path, const char* module_name) {
    // Create a loader for the serialized program.
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loaders_.insert(
        {module_name,
         std::make_unique<FileDataLoader>(std::move(loader.get()))});

    // Use it to load the program.
    Result<Program> program = Program::load(
        loaders_[module_name].get(),
        Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    programs_.insert(
        {module_name, std::make_unique<Program>(std::move(program.get()))});
  }

  void SetUp() override {
    load_program(std::getenv("ET_MODULE_ADD_PATH"), "add");
    load_program(std::getenv("ET_MODULE_STATEFUL_PATH"), "stateful");
  }

 private:
  // Must outlive program_, but tests shouldn't need to touch it.
  std::unordered_map<std::string, std::unique_ptr<FileDataLoader>> loaders_;

 protected:
  std::unordered_map<std::string, std::unique_ptr<Program>> programs_;
};

namespace {

// Check TensorInfo against hard coded values from AddModule.
void check_tensor(const TensorInfo& tensor_info) {
  auto sizes = tensor_info.sizes();
  auto dim_order = tensor_info.dim_order();
  EXPECT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 2);
  EXPECT_EQ(sizes[1], 2);
  EXPECT_EQ(tensor_info.scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(dim_order.size(), 2);
  EXPECT_EQ(dim_order[0], 0);
  EXPECT_EQ(dim_order[1], 1);
  EXPECT_EQ(tensor_info.is_memory_planned(), true);
  EXPECT_EQ(tensor_info.nbytes(), 16);
}
} // namespace

TEST_F(MethodMetaTest, MethodMetaApi) {
  Result<MethodMeta> method_meta = programs_["add"]->method_meta("forward");
  ASSERT_EQ(method_meta.error(), Error::Ok);

  // Appropriate amount of inputs
  EXPECT_EQ(method_meta->num_inputs(), 3);

  // Appropriate amount of outputs
  EXPECT_EQ(method_meta->num_outputs(), 1);

  // Appropriate amount of planned buffers
  EXPECT_EQ(method_meta->num_memory_planned_buffers(), 1);
  EXPECT_EQ(method_meta->num_non_const_buffers(), 1); // Deprecated API

  // Appropriate size of planned buffer
  EXPECT_EQ(method_meta->memory_planned_buffer_size(0).get(), 48);
  EXPECT_EQ(method_meta->non_const_buffer_size(0).get(), 48); // Deprecated API

  // Invalid index Errors
  EXPECT_EQ(
      method_meta->memory_planned_buffer_size(1).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      method_meta->non_const_buffer_size(1).error(),
      Error::InvalidArgument); // Deprecated API

  // Number instructions in method is nonzero
  EXPECT_NE(method_meta->num_instructions(), 0);

  // Missing method fails
  EXPECT_EQ(
      programs_["add"]->method_meta("not_a_method").error(),
      Error::InvalidArgument);
}

TEST_F(MethodMetaTest, TensorInfoApi) {
  Result<MethodMeta> method_meta = programs_["add"]->method_meta("forward");
  ASSERT_EQ(method_meta.error(), Error::Ok);

  // Input 1
  Result<TensorInfo> in_1 = method_meta->input_tensor_meta(0);
  ASSERT_TRUE(in_1.ok());
  check_tensor(in_1.get());

  // Input 2
  Result<TensorInfo> in_2 = method_meta->input_tensor_meta(1);
  ASSERT_TRUE(in_2.ok());
  check_tensor(in_2.get());

  // Output 1
  Result<TensorInfo> out_1 = method_meta->output_tensor_meta(0);
  ASSERT_TRUE(out_1.ok());
  check_tensor(out_1.get());

  // Copyable
  Result<TensorInfo> info = method_meta->input_tensor_meta(0);
  TensorInfo info_copy_ctor(info.get());
  TensorInfo info_copy_assign(out_1.get());
  info_copy_assign = info.get();
  check_tensor(info_copy_ctor);
  check_tensor(info_copy_assign);

  // Move-able
  TensorInfo info_move_ctor(std::move(info.get()));
  check_tensor(info_move_ctor);

  // Errors
  EXPECT_EQ(method_meta->input_tensor_meta(3).error(), Error::InvalidArgument);
  EXPECT_EQ(method_meta->input_tensor_meta(-1).error(), Error::InvalidArgument);
  EXPECT_EQ(method_meta->output_tensor_meta(3).error(), Error::InvalidArgument);
  EXPECT_EQ(
      method_meta->output_tensor_meta(-1).error(), Error::InvalidArgument);
}

TEST_F(MethodMetaTest, MethodMetaAttribute) {
  Result<MethodMeta> method_meta =
      programs_["stateful"]->method_meta("forward");
  ASSERT_EQ(method_meta.error(), Error::Ok);

  ASSERT_EQ(method_meta->num_attributes(), 1);
  auto state = method_meta->attribute_tensor_meta(0);
  ASSERT_TRUE(state.ok());

  ASSERT_EQ(state->name(), "state");
  ASSERT_FALSE(state->is_memory_planned());

  auto bad_access = method_meta->attribute_tensor_meta(1);
  ASSERT_EQ(bad_access.error(), Error::InvalidArgument);
}

TEST_F(MethodMetaTest, TensorInfoSizeOverflow) {
  // Create sizes that will cause overflow when multiplied
  std::vector<int32_t> overflow_sizes = {
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::max(),
  };

  // Create a minimal dim_order
  std::vector<uint8_t> dim_order = {0, 1, 2, 3};

  // Create a TensorInfo with the overflow sizes and expect it to fail.
  ET_EXPECT_DEATH(
      executorch::runtime::testing::TensorInfoTestFriend::get(
          Span<const int32_t>(overflow_sizes.data(), overflow_sizes.size()),
          Span<const uint8_t>(dim_order.data(), dim_order.size()),
          executorch::aten::ScalarType::Float,
          false, // is_memory_planned
          executorch::aten::string_view{nullptr, 0}),
      "");
}
