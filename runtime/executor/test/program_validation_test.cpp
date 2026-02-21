/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Program;
using executorch::runtime::Result;
using torch::executor::util::BufferDataLoader;
using torch::executor::util::FileDataLoader;

namespace {

// RAII wrapper for aligned buffer allocation.
class AlignedBuffer {
 public:
  explicit AlignedBuffer(const std::vector<uint8_t>& data)
      : buffer_(std::make_unique<uint8_t[]>(data.size() + kAlignment)),
        size_(data.size()) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer_.get());
    offset_ = (kAlignment - (addr % kAlignment)) % kAlignment;
    memcpy(buffer_.get() + offset_, data.data(), data.size());
  }

  uint8_t* data() {
    return buffer_.get() + offset_;
  }
  size_t size() const {
    return size_;
  }

  BufferDataLoader loader() {
    return BufferDataLoader(data(), size());
  }

 private:
  static constexpr size_t kAlignment = alignof(std::max_align_t);
  std::unique_ptr<uint8_t[]> buffer_;
  size_t offset_;
  size_t size_;
};

// EValue type configuration for program creation.
enum class EValueType { Tensor, Int, TensorList };

struct EValueConfig {
  EValueType type;
  std::vector<int32_t> tensor_sizes; // For Tensor type.
  std::vector<int32_t> tensor_list_items; // For TensorList type (indices).
};

// Unified helper to create a minimal valid PTE flatbuffer with configurable
// evalues. Returns a buffer containing the flatbuffer data.
std::vector<uint8_t> CreateTestProgram(
    const std::vector<EValueConfig>& configs) {
  flatbuffers::FlatBufferBuilder builder(1024);

  std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>> evalues;

  for (const auto& config : configs) {
    switch (config.type) {
      case EValueType::Tensor: {
        auto sizes_vec = builder.CreateVector(config.tensor_sizes);
        std::vector<uint8_t> dim_order(config.tensor_sizes.size());
        for (size_t i = 0; i < config.tensor_sizes.size(); i++) {
          dim_order[i] = static_cast<uint8_t>(i);
        }
        auto dim_order_vec = builder.CreateVector(dim_order);
        auto tensor = executorch_flatbuffer::CreateTensor(
            builder,
            executorch_flatbuffer::ScalarType::FLOAT,
            /*storage_offset=*/0,
            sizes_vec,
            dim_order_vec,
            /*requires_grad=*/false,
            /*data_buffer_idx=*/0,
            /*allocation_info=*/0,
            /*layout=*/0,
            executorch_flatbuffer::TensorShapeDynamism::STATIC,
            /*extra_tensor_info=*/0);
        evalues.push_back(executorch_flatbuffer::CreateEValue(
            builder,
            executorch_flatbuffer::KernelTypes::Tensor,
            tensor.Union()));
        break;
      }
      case EValueType::Int: {
        auto int_val = executorch_flatbuffer::CreateInt(builder, 42);
        evalues.push_back(executorch_flatbuffer::CreateEValue(
            builder, executorch_flatbuffer::KernelTypes::Int, int_val.Union()));
        break;
      }
      case EValueType::TensorList: {
        auto items = builder.CreateVector(config.tensor_list_items);
        auto tensor_list =
            executorch_flatbuffer::CreateTensorList(builder, items);
        evalues.push_back(executorch_flatbuffer::CreateEValue(
            builder,
            executorch_flatbuffer::KernelTypes::TensorList,
            tensor_list.Union()));
        break;
      }
      default:
        ET_CHECK_MSG(false, "Invalid EValueType");
    }
  }

  auto values_vec = builder.CreateVector(evalues);
  auto plan_name = builder.CreateString("forward");
  auto empty_int_vec = builder.CreateVector(std::vector<int32_t>{});
  auto empty_int64_vec = builder.CreateVector(std::vector<int64_t>{0});
  auto empty_chain_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  auto empty_operators_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  auto empty_delegates_vec = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});

  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      /*container_meta_type=*/0,
      values_vec,
      empty_int_vec,
      empty_int_vec,
      empty_chain_vec,
      empty_operators_vec,
      empty_delegates_vec,
      empty_int64_vec);

  std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>> plans;
  plans.push_back(execution_plan);
  auto plans_vec = builder.CreateVector(plans);

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      /*version=*/0,
      plans_vec,
      /*constant_buffer=*/0,
      /*backend_delegate_data=*/0,
      /*segments=*/0,
      /*constant_segment=*/0,
      /*mutable_data_segments=*/0,
      /*named_data=*/0);

  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  const uint8_t* buf = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  return std::vector<uint8_t>(buf, buf + size);
}

} // namespace

class ProgramValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();

    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    add_loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));
  }

  std::unique_ptr<FileDataLoader> add_loader_;
};

TEST_F(ProgramValidationTest, ValidProgramPassesInternalConsistency) {
  Result<Program> program = Program::load(
      add_loader_.get(), Program::Verification::InternalConsistency);
  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramValidationTest, InternalConsistencyDetectsTruncatedData) {
  size_t full_data_len = add_loader_->size().get();
  Result<FreeableBuffer> full_data = add_loader_->load(
      /*offset=*/0,
      full_data_len,
      executorch::runtime::DataLoader::SegmentInfo(
          executorch::runtime::DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(full_data.error(), Error::Ok);

  BufferDataLoader half_data_loader(full_data->data(), full_data_len / 2);

  Result<Program> program = Program::load(
      &half_data_loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramValidationTest, TensorNumelOverflowDetected) {
  std::vector<EValueConfig> configs = {
      {EValueType::Tensor, {2000000000, 2000000000, 2000000000}, {}}};

  AlignedBuffer buf(CreateTestProgram(configs));
  auto loader = buf.loader();

  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  EXPECT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramValidationTest, TensorNumelOverflowNotDetectedWithMinimal) {
  std::vector<EValueConfig> configs = {
      {EValueType::Tensor, {2000000000, 2000000000, 2000000000}, {}}};

  AlignedBuffer buf(CreateTestProgram(configs));
  auto loader = buf.loader();

  // Minimal verification doesn't run program validation.
  Result<Program> program =
      Program::load(&loader, Program::Verification::Minimal);
}

TEST_F(ProgramValidationTest, NegativeSizeDetected) {
  std::vector<EValueConfig> configs = {{EValueType::Tensor, {10, -5, 10}, {}}};

  AlignedBuffer buf(CreateTestProgram(configs));
  auto loader = buf.loader();

  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  EXPECT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramValidationTest, TensorListWithIntElementDetected) {
  // values[0] = Tensor, values[1] = Int (INVALID!), values[2] = TensorList([0,
  // 1])
  std::vector<EValueConfig> configs = {
      {EValueType::Tensor, {2, 3}, {}},
      {EValueType::Int, {}, {}},
      {EValueType::TensorList, {}, {0, 1}}};

  AlignedBuffer buf(CreateTestProgram(configs));
  auto loader = buf.loader();

  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  EXPECT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramValidationTest, TensorListWithOutOfBoundsIndexDetected) {
  // values[0] = Tensor, values[1] = TensorList([0, 99]) - 99 is out of bounds
  std::vector<EValueConfig> configs = {
      {EValueType::Tensor, {2, 3}, {}}, {EValueType::TensorList, {}, {0, 99}}};

  AlignedBuffer buf(CreateTestProgram(configs));
  auto loader = buf.loader();

  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  EXPECT_EQ(program.error(), Error::InvalidProgram);
}
