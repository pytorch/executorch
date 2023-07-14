#include <cctype>
#include <filesystem>

#include <cstring>
#include <memory>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <executorch/util/embedded_data_loader.h>
#include <executorch/util/file_data_loader.h>

#include <gtest/gtest.h>

using namespace ::testing;
using torch::executor::Error;
using torch::executor::FreeableBuffer;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::util::EmbeddedDataLoader;
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
    torch::executor::runtime_init();

    // Load the serialized ModuleAdd data.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::From(path);
    ASSERT_EQ(loader.error(), Error::Ok);

    // This file should always be compatible.
    Result<FreeableBuffer> header =
        loader->Load(/*offset=*/0, Program::kMinHeadBytes);
    ASSERT_EQ(header.error(), Error::Ok);
    EXPECT_EQ(
        Program::check_header(header->data(), header->size()),
        Program::HeaderStatus::CompatibleVersion);

    add_loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Load the serialized ModuleAdd data.
    path = std::getenv("ET_MODULE_MULTI_ENTRY_PATH");
    Result<FileDataLoader> multi_loader = FileDataLoader::From(path);
    ASSERT_EQ(multi_loader.error(), Error::Ok);

    // This file should always be compatible.
    Result<FreeableBuffer> multi_header =
        multi_loader->Load(/*offset=*/0, Program::kMinHeadBytes);
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

namespace torch {
namespace executor {
namespace testing {
// Provides access to private Program methods.
class ProgramTestFriend final {
 public:
  __ET_NODISCARD static Result<FreeableBuffer> LoadSegment(
      const Program* program,
      size_t index) {
    return program->LoadSegment(index);
  }
};
} // namespace testing
} // namespace executor
} // namespace torch

using torch::executor::testing::ProgramTestFriend;

TEST_F(ProgramTest, DataParsesWithMinimalVerification) {
  // Parse the Program from the data.
  Result<Program> program =
      Program::Load(add_loader_.get(), Program::Verification::Minimal);

  // Should have succeeded.
  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramTest, DataParsesWithInternalConsistencyVerification) {
  // Parse the Program from the data.
  Result<Program> program = Program::Load(
      add_loader_.get(), Program::Verification::InternalConsistency);

  // Should have succeeded.
  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramTest, BadMagicFailsToLoad) {
  // Make a local copy of the data.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
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
  EmbeddedDataLoader data_loader(data.get(), data_len);

  {
    // Parse the Program from the data. Use minimal verification to show that
    // even this catches the header problem.
    Result<Program> program =
        Program::Load(&data_loader, Program::Verification::Minimal);

    // Should fail.
    ASSERT_EQ(program.error(), Error::InvalidProgram);
  }

  // Fix the data.
  data[4] = 'E';
  data[5] = 'T';

  {
    // Parse the Program from the data again.
    Result<Program> program =
        Program::Load(&data_loader, Program::Verification::Minimal);

    // Should now succeed.
    ASSERT_EQ(program.error(), Error::Ok);
  }
}

TEST_F(ProgramTest, VerificationCatchesTruncation) {
  // Get the program data.
  size_t full_data_len = add_loader_->size().get();
  Result<FreeableBuffer> full_data =
      add_loader_->Load(/*offset=*/0, full_data_len);
  ASSERT_EQ(full_data.error(), Error::Ok);

  // Make a loader that only exposes half of the data.
  EmbeddedDataLoader half_data_loader(full_data->data(), full_data_len / 2);

  // Loading with full verification should fail.
  Result<Program> program = Program::Load(
      &half_data_loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, VerificationCatchesCorruption) {
  // Make a local copy of the data.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get(), src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Corrupt the second half of the data.
  std::memset(&data[data_len / 2], 0x55, data_len - (data_len / 2));

  // Wrap the corrupted data in a loader.
  EmbeddedDataLoader data_loader(data.get(), data_len);

  // Should fail to parse corrupted data when using full verification.
  Result<Program> program =
      Program::Load(&data_loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, UnalignedProgramDataFails) {
  // Make a local copy of the data, on an odd alignment.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len + 1);
  {
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get() + 1, src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Wrap the offset data in a loader.
  EmbeddedDataLoader data_loader(data.get() + 1, data_len);

  // Should refuse to accept unaligned data.
  Result<Program> program =
      Program::Load(&data_loader, Program::Verification::Minimal);
  ASSERT_NE(program.error(), Error::Ok);
}

TEST_F(ProgramTest, LoadSegmentWithNoSegments) {
  // Load a program with no segments.
  Result<Program> program =
      Program::Load(add_loader_.get(), kDefaultVerification);
  EXPECT_EQ(program.error(), Error::Ok);

  // Loading a segment should fail.
  Result<FreeableBuffer> segment =
      ProgramTestFriend::LoadSegment(&program.get(), 0);
  EXPECT_NE(segment.error(), Error::Ok);
}

TEST_F(ProgramTest, ShortDataHeader) {
  Result<FreeableBuffer> header =
      add_loader_->Load(/*offset=*/0, Program::kMinHeadBytes);
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
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
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
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
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

// TODO(T144120904): Add tests for programs with segments once we can generate
// them.

TEST_F(ProgramTest, DEPRECATEDDataParsesAndIsValid) {
  // Load the whole file into a FreeableBuffer.
  size_t data_len = add_loader_->size().get();
  Result<FreeableBuffer> data = add_loader_->Load(/*offset=*/0, data_len);
  ASSERT_EQ(data.error(), Error::Ok);
  ASSERT_EQ(data->size(), data_len);

  // Parse the Program from the data.
  Program program(data->data());

  // Should be valid.
  EXPECT_TRUE(program.is_valid());

  // Method calls should succeed without hitting ET_CHECK.
  program.get_constant_buffer_data(0);
  program.constant_buffer_size();
  program.get_non_const_buffer_size(1, "forward");
  auto res = program.num_non_const_buffers("forward");
  EXPECT_TRUE(res.ok());
  EXPECT_EQ(res.get(), 2);
}

TEST_F(ProgramTest, DEPRECATEDBadMagicIsInvalid) {
  // Make a local copy of the data.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len);
  {
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
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

  // Parse the Program from the data.
  Program program(data.get());

  // Should not be valid.
  EXPECT_FALSE(program.is_valid());

  // Method calls should die from ET_CHECK.
  ET_EXPECT_DEATH(program.get_constant_buffer_data(0), "");
  ET_EXPECT_DEATH(program.constant_buffer_size(), "");
  ET_EXPECT_DEATH(program.get_non_const_buffer_size(1), "");
  ET_EXPECT_DEATH(program.num_non_const_buffers(), "");
}

TEST_F(ProgramTest, DEPRECATEDUnalignedProgramDataFails) {
  // Make a local copy of the data, on an odd alignment.
  size_t data_len = add_loader_->size().get();
  auto data = std::make_unique<char[]>(data_len + 1);
  {
    Result<FreeableBuffer> src = add_loader_->Load(/*offset=*/0, data_len);
    ASSERT_EQ(src.error(), Error::Ok);
    ASSERT_EQ(src->size(), data_len);
    memcpy(data.get() + 1, src->data(), data_len);
    // FreeableBuffer goes out of scope and frees its data.
  }

  // Parse the Program from the data.
  Program program(data.get());

  // Should not be valid.
  EXPECT_FALSE(program.is_valid());
}

TEST_F(ProgramTest, getMethods) {
  // Parse the Program from the data.
  Result<Program> program_res =
      Program::Load(multi_loader_.get(), kDefaultVerification);
  EXPECT_EQ(program_res.error(), Error::Ok);

  Program program(std::move(program_res.get()));

  // Should not be valid.
  EXPECT_TRUE(program.is_valid());

  // Method calls should succeed without hitting ET_CHECK.
  EXPECT_EQ(program.num_methods(), 2);
  auto res = program.get_method_name(0);
  EXPECT_TRUE(res.ok());
  EXPECT_EQ(strcmp(res.get(), "forward"), 0);
  auto res2 = program.get_method_name(1);
  EXPECT_TRUE(res2.ok());
  EXPECT_EQ(strcmp(res2.get(), "forward2"), 0);
}
