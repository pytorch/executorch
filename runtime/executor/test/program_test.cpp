/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/program.h>

#include <cctype>
#include <filesystem>

#include <cstring>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Program;
using executorch::runtime::Result;
using torch::executor::util::BufferDataLoader;
using torch::executor::util::FileDataLoader;

// Verification level to use for tests not specifically focused on verification.
// Use the highest level to exercise it more.
constexpr Program::Verification kDefaultVerification =
    Program::Verification::InternalConsistency;

class ProgramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Load the serialized ModuleAdd data.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);

    // This file should always be compatible.
    Result<FreeableBuffer> header = loader->load(
        /*offset=*/0,
        Program::kMinHeadBytes,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(header.error(), Error::Ok);
    EXPECT_EQ(
        Program::check_header(header->data(), header->size()),
        Program::HeaderStatus::CompatibleVersion);

    add_loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Load the serialized ModuleAdd data.
    path = std::getenv("ET_MODULE_MULTI_ENTRY_PATH");
    Result<FileDataLoader> multi_loader = FileDataLoader::from(path);
    ASSERT_EQ(multi_loader.error(), Error::Ok);

    // This file should always be compatible.
    Result<FreeableBuffer> multi_header = multi_loader->load(
        /*offset=*/0,
        Program::kMinHeadBytes,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(multi_header.error(), Error::Ok);
    EXPECT_EQ(
        Program::check_header(multi_header->data(), multi_header->size()),
        Program::HeaderStatus::CompatibleVersion);

    multi_loader_ =
        std::make_unique<FileDataLoader>(std::move(multi_loader.get()));
  }

  std::unique_ptr<FileDataLoader> add_loader_;
  std::unique_ptr<FileDataLoader> multi_loader_;
};

namespace executorch {
namespace runtime {
namespace testing {
// Provides access to private Program methods.
class ProgramTestFriend final {
 public:
  __ET_NODISCARD static Result<FreeableBuffer> LoadSegment(
      const Program* program,
      const DataLoader::SegmentInfo& segment_info) {
    return program->LoadSegment(segment_info);
  }

  const static executorch_flatbuffer::Program* GetInternalProgram(
      const Program* program) {
    return program->internal_program_;
  }
};
} // namespace testing
} // namespace runtime
} // namespace executorch

using executorch::runtime::testing::ProgramTestFriend;

TEST_F(ProgramTest, DataParsesWithMinimalVerification) {
  // Parse the Program from the data.
  Result<Program> program =
      Program::load(add_loader_.get(), Program::Verification::Minimal);

  // Should have succeeded.
  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramTest, DataParsesWithInternalConsistencyVerification) {
  // Parse the Program from the data.
  Result<Program> program = Program::load(
      add_loader_.get(), Program::Verification::InternalConsistency);

  // Should have succeeded.
  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramTest, BadMagicFailsToLoad) {
  // Make a local copy of the data.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->load(
        /*offset=*/0,
        data_len,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get(), src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Corrupt the magic value.
  EXPECT_EQ(data[4], 'E');
  data[4] = 'X';
  EXPECT_EQ(data[5], 'T');
  data[5] = 'Y';

  // Wrap the modified data in a loader.
  BufferDataLoader data_loader(data.get(), data_len);

  {
    // Parse the Program from the data. Use minimal verification to show that
    // even this catches the header problem.
    Result<Program> program =
        Program::load(&data_loader, Program::Verification::Minimal);

    // Should fail.
    ASSERT_EQ(program.error(), Error::InvalidProgram);
  }

  // Fix the data.
  data[4] = 'E';
  data[5] = 'T';

  {
    // Parse the Program from the data again.
    Result<Program> program =
        Program::load(&data_loader, Program::Verification::Minimal);

    // Should now succeed.
    ASSERT_EQ(program.error(), Error::Ok);
  }
}

TEST_F(ProgramTest, VerificationCatchesTruncation) {
  // Get the program data.
  size_t full_data_len = add_loader_->size().get();
  Result<FreeableBuffer> full_data = add_loader_->load(
      /*offset=*/0,
      full_data_len,
      /*segment_info=*/
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(full_data.error(), Error::Ok);

  // Make a loader that only exposes half of the data.
  BufferDataLoader half_data_loader(full_data->data(), full_data_len / 2);

  // Loading with full verification should fail.
  Result<Program> program = Program::load(
      &half_data_loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, VerificationCatchesCorruption) {
  // Make a local copy of the data.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->load(
        /*offset=*/0,
        data_len,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get(), src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Corrupt the second half of the data.
  std::memset(&data[data_len / 2], 0x55, data_len - (data_len / 2));

  // Wrap the corrupted data in a loader.
  BufferDataLoader data_loader(data.get(), data_len);

  // Should fail to parse corrupted data when using full verification.
  Result<Program> program =
      Program::load(&data_loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, UnalignedProgramDataFails) {
  // Make a local copy of the data, on an odd alignment.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len + 1);
  {
    Result<FreeableBuffer> src = add_loader_->load(
        /*offset=*/0,
        data_len,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get() + 1, src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Wrap the offset data in a loader.
  BufferDataLoader data_loader(data.get() + 1, data_len);

  // Should refuse to accept unaligned data.
  Result<Program> program =
      Program::load(&data_loader, Program::Verification::Minimal);
  ASSERT_NE(program.error(), Error::Ok);
}

TEST_F(ProgramTest, LoadSegmentWithNoSegments) {
  // Load a program with no appended segments.
  Result<Program> program =
      Program::load(add_loader_.get(), kDefaultVerification);
  EXPECT_EQ(program.error(), Error::Ok);

  // Loading a non-program segment should fail.
  const auto segment_info = DataLoader::SegmentInfo(
      DataLoader::SegmentInfo::Type::Backend,
      /*segment_index=*/0,
      "some-backend");
  Result<FreeableBuffer> segment =
      ProgramTestFriend::LoadSegment(&program.get(), segment_info);
  EXPECT_NE(segment.error(), Error::Ok);
}

TEST_F(ProgramTest, ShortDataHeader) {
  Result<FreeableBuffer> header = add_loader_->load(
      /*offset=*/0,
      Program::kMinHeadBytes,
      /*segment_info=*/
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(header.error(), Error::Ok);

  // Provide less than the required amount of data.
  EXPECT_EQ(
      Program::check_header(header->data(), Program::kMinHeadBytes - 1),
      Program::HeaderStatus::ShortData);
}

TEST_F(ProgramTest, IncompatibleHeader) {
  // Make a local copy of the header.
  size_t data_len = Program::kMinHeadBytes;
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->load(
        /*offset=*/0,
        data_len,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get(), src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Change the number part of the magic value to a different value.
  EXPECT_EQ(data[4], 'E');
  EXPECT_EQ(data[5], 'T');
  EXPECT_TRUE(std::isdigit(data[6])) << "Not a digit: " << data[6];
  EXPECT_TRUE(std::isdigit(data[7])) << "Not a digit: " << data[7];

  // Modify the tens digit.
  if (data[6] == '9') {
    data[6] = '0';
  } else {
    data[6] += 1;
  }
  EXPECT_TRUE(std::isdigit(data[6])) << "Not a digit: " << data[6];

  // Should count as present but incompatible.
  EXPECT_EQ(
      Program::check_header(data.get(), data_len),
      Program::HeaderStatus::IncompatibleVersion);
}

TEST_F(ProgramTest, HeaderNotPresent) {
  // Make a local copy of the header.
  size_t data_len = Program::kMinHeadBytes;
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->load(
        /*offset=*/0,
        data_len,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get(), src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Corrupt the magic value.
  EXPECT_EQ(data[4], 'E');
  data[4] = 'X';
  EXPECT_EQ(data[5], 'T');
  data[5] = 'Y';

  // The header is not present.
  EXPECT_EQ(
      Program::check_header(data.get(), data_len),
      Program::HeaderStatus::NotPresent);
}

TEST_F(ProgramTest, getMethods) {
  // Parse the Program from the data.
  Result<Program> program_res =
      Program::load(multi_loader_.get(), kDefaultVerification);
  EXPECT_EQ(program_res.error(), Error::Ok);

  Program program(std::move(program_res.get()));

  // Method calls should succeed without hitting ET_CHECK.
  EXPECT_EQ(program.num_methods(), 2);
  auto res = program.get_method_name(0);
  EXPECT_TRUE(res.ok());
  EXPECT_EQ(strcmp(res.get(), "forward"), 0);
  auto res2 = program.get_method_name(1);
  EXPECT_TRUE(res2.ok());
  EXPECT_EQ(strcmp(res2.get(), "forward2"), 0);
}

// Test that the deprecated Load method (capital 'L') still works.
TEST_F(ProgramTest, DEPRECATEDLoad) {
  // Parse the Program from the data.
  // NOLINTNEXTLINE(facebook-hte-Deprecated)
  Result<Program> program_res = Program::Load(multi_loader_.get());
  EXPECT_EQ(program_res.error(), Error::Ok);
}

TEST_F(ProgramTest, LoadConstantSegment) {
  // Load the serialized ModuleLinear data, with constants in the segment and no
  // constants in the flatbuffer.
  const char* linear_path =
      std::getenv("ET_MODULE_LINEAR_CONSTANT_SEGMENT_PATH");
  Result<FileDataLoader> linear_loader = FileDataLoader::from(linear_path);
  ASSERT_EQ(linear_loader.error(), Error::Ok);

  // This file should always be compatible.
  Result<FreeableBuffer> linear_header = linear_loader->load(
      /*offset=*/0,
      Program::kMinHeadBytes,
      /*segment_info=*/
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(linear_header.error(), Error::Ok);
  EXPECT_EQ(
      Program::check_header(linear_header->data(), linear_header->size()),
      Program::HeaderStatus::CompatibleVersion);

  Result<Program> program = Program::load(&linear_loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  // Load constant segment data, which is currently always in segment index
  // zero.
  const auto segment_info = DataLoader::SegmentInfo(
      DataLoader::SegmentInfo::Type::Constant,
      /*segment_index=*/0);
  Result<FreeableBuffer> segment =
      ProgramTestFriend::LoadSegment(&program.get(), segment_info);
  EXPECT_EQ(segment.error(), Error::Ok);

  const executorch_flatbuffer::Program* flatbuffer_program =
      ProgramTestFriend::GetInternalProgram(&program.get());

  // Expect one segment containing the constants.
  EXPECT_EQ(flatbuffer_program->segments()->size(), 1);

  // The constant buffer should be empty.
  EXPECT_EQ(flatbuffer_program->constant_buffer()->size(), 0);

  // Check constant segment offsets.
  EXPECT_EQ(flatbuffer_program->constant_segment()->segment_index(), 0);
  EXPECT_GE(flatbuffer_program->constant_segment()->offsets()->size(), 1);
}

TEST_F(ProgramTest, LoadConstantSegmentWithNoConstantSegment) {
  // Load the serialized ModuleLinear data, with constants in the flatbuffer and
  // no constants in the segment.
  const char* linear_path =
      std::getenv("ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH");
  Result<FileDataLoader> linear_loader = FileDataLoader::from(linear_path);
  ASSERT_EQ(linear_loader.error(), Error::Ok);

  // This file should always be compatible.
  Result<FreeableBuffer> linear_header = linear_loader->load(
      /*offset=*/0,
      Program::kMinHeadBytes,
      /*segment_info=*/
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(linear_header.error(), Error::Ok);
  EXPECT_EQ(
      Program::check_header(linear_header->data(), linear_header->size()),
      Program::HeaderStatus::CompatibleVersion);

  Result<Program> program = Program::load(&linear_loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  const executorch_flatbuffer::Program* flatbuffer_program =
      ProgramTestFriend::GetInternalProgram(&program.get());

  // Expect no segments.
  EXPECT_EQ(flatbuffer_program->segments()->size(), 0);

  // The constant buffer should exist.
  EXPECT_GE(flatbuffer_program->constant_buffer()->size(), 1);
}
