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
#include <fstream>
#include <system_error>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
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

    // Load the serialized ModuleMultiEntry data.
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
  ET_NODISCARD static Result<FreeableBuffer> LoadSegment(
      const Program* program,
      const DataLoader::SegmentInfo& segment_info) {
    return program->LoadSegment(segment_info);
  }

  ET_NODISCARD static Error load_mutable_subsegment_into(
      const Program* program,
      size_t mutable_data_segments_index,
      size_t offset_index,
      size_t size,
      void* buffer) {
    return program->load_mutable_subsegment_into(
        mutable_data_segments_index, offset_index, size, buffer);
  }

  ET_NODISCARD static Result<FreeableBuffer> load_mutable_subsegment(
      const Program* program,
      size_t mutable_data_segments_index,
      size_t offset_index,
      size_t size) {
    return program->load_mutable_subsegment(
        mutable_data_segments_index, offset_index, size);
  }

  const static executorch_flatbuffer::Program* GetInternalProgram(
      const Program* program) {
    return program->internal_program_;
  }
  static Result<const void*> get_constant_buffer_data(
      const Program* program,
      size_t buffer_index,
      size_t nbytes) {
    return program->get_constant_buffer_data(buffer_index, nbytes);
  }

  // Constructs a Program directly with a chosen segment_base_offset and a
  // pre-built FlatBuffer body. Used to set up malformed states (e.g.
  // segment_base_offset != 0 but segments == null) that Program::load cannot
  // produce, since segment_base_offset is driven only by the extended header.
  static Program MakeProgram(
      DataLoader* loader,
      size_t segment_base_offset,
      const executorch_flatbuffer::Program* internal_program) {
    return Program(
        loader,
        segment_base_offset,
        FreeableBuffer{},
        internal_program,
        FreeableBuffer{},
        std::nullopt);
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

namespace {

// Builds the smallest program that Program::load() accepts, stamped with the
// given schema version.
std::vector<uint8_t> CreateProgramWithVersion(uint32_t version) {
  flatbuffers::FlatBufferBuilder builder(1024);

  auto plan_name = builder.CreateString("forward");
  auto empty_values = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{});
  auto empty_inputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_outputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  auto empty_operators = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  auto empty_delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  auto buffer_sizes = builder.CreateVector(std::vector<int64_t>{0});

  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      /*container_meta_type=*/0,
      empty_values,
      empty_inputs,
      empty_outputs,
      empty_chains,
      empty_operators,
      empty_delegates,
      buffer_sizes);
  auto execution_plans = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
          execution_plan});

  // A constant segment holding only the placeholder offset means "no
  // constants", which keeps this program off the deprecated constant_buffer
  // path that OSS builds compile out.
  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      version,
      execution_plans,
      /*constant_buffer=*/0,
      /*backend_delegate_data=*/0,
      /*segments=*/0,
      constant_segment);
  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  const uint8_t* data = builder.GetBufferPointer();
  return std::vector<uint8_t>(data, data + builder.GetSize());
}

// Builds a program whose constant segment exists but holds no bytes. A program
// whose only constants are zero-length tensors looks exactly like this: the
// offsets vector has a real entry per constant, while the segment itself is
// empty.
std::vector<uint8_t> CreateProgramWithEmptyConstantSegment() {
  flatbuffers::FlatBufferBuilder builder(1024);

  auto plan_name = builder.CreateString("forward");
  auto empty_values = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{});
  auto empty_inputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_outputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  auto empty_operators = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  auto empty_delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  auto buffer_sizes = builder.CreateVector(std::vector<int64_t>{0});

  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      /*container_meta_type=*/0,
      empty_values,
      empty_inputs,
      empty_outputs,
      empty_chains,
      empty_operators,
      empty_delegates,
      buffer_sizes);
  auto execution_plans = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
          execution_plan});

  // Index 0 is the usual non-constant placeholder; index 1 is a zero-length
  // constant, which occupies no bytes in the segment.
  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0, 0}));
  auto segments = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{
          executorch_flatbuffer::CreateDataSegment(
              builder, /*offset=*/0, /*size=*/0)});

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      Program::kMaxSupportedSchemaVersion,
      execution_plans,
      /*constant_buffer=*/0,
      /*backend_delegate_data=*/0,
      segments,
      constant_segment);
  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  const uint8_t* data = builder.GetBufferPointer();
  return std::vector<uint8_t>(data, data + builder.GetSize());
}

} // namespace

// An empty constant segment loads to a null pointer, which must not be mistaken
// for "this program uses the deprecated inline constant_buffer". Doing so fails
// the load outright in builds that compile that path out.
TEST_F(ProgramTest, EmptyConstantSegmentIsStillASegment) {
  std::vector<uint8_t> data = CreateProgramWithEmptyConstantSegment();

  // Must go through a loader that returns a null pointer for a zero-length
  // read, which is what FileDataLoader does ("don't bother allocating for
  // empty segments") and what makes the segment indistinguishable from an
  // absent one. BufferDataLoader hands back a pointer into its own buffer and
  // would hide this.
  // std::filesystem::path, not std::string: path::value_type is wchar_t on
  // Windows, so there is no implicit conversion to std::string there.
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
      "et_empty_constant_segment_test.pte";
  {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
  }

  // Declared before the loader so that it runs after the loader has closed the
  // file: Windows refuses to delete a file that still has an open handle. The
  // non-throwing overload keeps a failed cleanup from failing the test.
  struct RemoveOnExit {
    const std::filesystem::path& path;
    ~RemoveOnExit() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } remove_on_exit{path};

  Result<FileDataLoader> loader = FileDataLoader::from(path.string().c_str());
  ASSERT_EQ(loader.error(), Error::Ok);
  Result<Program> program = Program::load(&loader.get(), kDefaultVerification);
  ASSERT_EQ(program.error(), Error::Ok);

  // The zero-length constant at index 1 resolves, rather than being rejected.
  Result<const void*> constant_data =
      program->get_constant_buffer_data(/*buffer_idx=*/1, /*nbytes=*/0);
  EXPECT_EQ(constant_data.error(), Error::Ok);
}

TEST_F(ProgramTest, SupportedSchemaVersionLoads) {
  std::vector<uint8_t> data =
      CreateProgramWithVersion(Program::kMaxSupportedSchemaVersion);

  alignas(16) uint8_t aligned_buffer[2048];
  ASSERT_LE(data.size(), sizeof(aligned_buffer));
  memcpy(aligned_buffer, data.data(), data.size());

  BufferDataLoader data_loader(aligned_buffer, data.size());
  Result<Program> program = Program::load(&data_loader, kDefaultVerification);

  EXPECT_EQ(program.error(), Error::Ok);
}

TEST_F(ProgramTest, NewerSchemaVersionFailsToLoad) {
  std::vector<uint8_t> data =
      CreateProgramWithVersion(Program::kMaxSupportedSchemaVersion + 1);

  alignas(16) uint8_t aligned_buffer[2048];
  ASSERT_LE(data.size(), sizeof(aligned_buffer));
  memcpy(aligned_buffer, data.data(), data.size());

  // Use minimal verification to show that even the cheapest level catches it.
  BufferDataLoader data_loader(aligned_buffer, data.size());
  Result<Program> program =
      Program::load(&data_loader, Program::Verification::Minimal);

  EXPECT_EQ(program.error(), Error::InvalidProgram);
}

// These tests require ET_ENABLE_PROGRAM_VERIFICATION to be enabled.
// In Release builds, verification is disabled by default to save binary size.
#ifndef ET_ENABLE_PROGRAM_VERIFICATION
#define ET_ENABLE_PROGRAM_VERIFICATION 1
#endif

// ET_ENABLE_DEPRECATED_CONSTANT_BUFFER is 1 by default, and
// set to 0 in the root CMakeLists.txt for OSS builds.
#ifndef ET_ENABLE_DEPRECATED_CONSTANT_BUFFER
#define ET_ENABLE_DEPRECATED_CONSTANT_BUFFER 1
#endif

#if ET_ENABLE_PROGRAM_VERIFICATION

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

#endif // ET_ENABLE_PROGRAM_VERIFICATION

TEST_F(ProgramTest, MinimalVerificationCatchesInvalidRootOffset) {
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
  }

  // Corrupt the root offset (first 4 bytes) to point beyond the buffer.
  // Use WriteScalar for correct little-endian encoding.
  flatbuffers::WriteScalar<flatbuffers::uoffset_t>(
      data.get(), static_cast<flatbuffers::uoffset_t>(data_len + 1000));

  // Wrap the corrupted data in a loader.
  BufferDataLoader data_loader(data.get(), data_len);

  // Should fail with Minimal verification due to invalid root offset.
  Result<Program> program =
      Program::load(&data_loader, Program::Verification::Minimal);
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

TEST_F(ProgramTest, LoadSegmentWithNullSegmentsDoesNotCrash) {
  // A non-zero segment_base_offset with an absent `segments` table must return
  // InvalidProgram rather than dereferencing null.
  flatbuffers::FlatBufferBuilder builder(256);
  builder.Finish(
      executorch_flatbuffer::CreateProgram(builder),
      executorch_flatbuffer::ProgramIdentifier());
  const auto* internal_program =
      executorch_flatbuffer::GetProgram(builder.GetBufferPointer());

  uint8_t dummy[16] = {};
  BufferDataLoader loader(dummy, sizeof(dummy));
  Program program =
      ProgramTestFriend::MakeProgram(&loader, 16, internal_program);

  Result<FreeableBuffer> result = ProgramTestFriend::LoadSegment(
      &program,
      DataLoader::SegmentInfo(
          DataLoader::SegmentInfo::Type::Backend, /*segment_index=*/0, "b"));
  EXPECT_EQ(result.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, LoadMutableSubsegmentWithNullSegmentsDoesNotCrash) {
  // Same malformed state reached through load_mutable_subsegment_into:
  // mutable_data_segments is populated so the function passes its own guards,
  // but segments is absent.
  flatbuffers::FlatBufferBuilder builder(256);
  auto subsegment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));
  builder.Finish(
      executorch_flatbuffer::CreateProgram(
          builder,
          /*version=*/0,
          /*execution_plan=*/0,
          /*constant_buffer=*/0,
          /*backend_delegate_data=*/0,
          /*segments=*/0,
          /*constant_segment=*/0,
          builder.CreateVector(
              std::vector<flatbuffers::Offset<
                  executorch_flatbuffer::SubsegmentOffsets>>{subsegment})),
      executorch_flatbuffer::ProgramIdentifier());
  const auto* internal_program =
      executorch_flatbuffer::GetProgram(builder.GetBufferPointer());

  uint8_t dummy[16] = {};
  BufferDataLoader loader(dummy, sizeof(dummy));
  Program program =
      ProgramTestFriend::MakeProgram(&loader, 16, internal_program);

  uint8_t out[4] = {};
  EXPECT_EQ(
      ProgramTestFriend::load_mutable_subsegment_into(
          &program,
          /*mutable_data_segments_index=*/0,
          /*offset_index=*/0,
          sizeof(out),
          out),
      Error::InvalidProgram);
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

TEST_F(ProgramTest, GetNamedDataMap_Fail) {
  Result<Program> program =
      Program::load(add_loader_.get(), kDefaultVerification);
  ASSERT_EQ(program.error(), Error::Ok);

  // Get the named data map. Expect to fail, as add.pte does not have any
  // named data segments.
  Result<const executorch::runtime::NamedDataMap*> named_data_map =
      program->get_named_data_map();
  EXPECT_EQ(named_data_map.error(), Error::NotFound);
}

// Test that the deprecated Load method (capital 'L') still works.
TEST_F(ProgramTest, DEPRECATEDLoad) {
  // Parse the Program from the data.
  // NOLINTNEXTLINE(facebook-hte-Deprecated)
  Result<Program> program_res = Program::Load(multi_loader_.get());
  EXPECT_EQ(program_res.error(), Error::Ok);
}

TEST_F(ProgramTest, LoadConstantSegmentWithNoConstantSegment) {
  Result<Program> program =
      Program::load(add_loader_.get(), kDefaultVerification);
  ASSERT_EQ(program.error(), Error::Ok);

  // Load constant segment data should fail.
  const auto segment_info = DataLoader::SegmentInfo(
      DataLoader::SegmentInfo::Type::Constant,
      /*segment_index=*/0);
  Result<FreeableBuffer> segment =
      ProgramTestFriend::LoadSegment(&program.get(), segment_info);
  EXPECT_NE(segment.error(), Error::Ok);

  const executorch_flatbuffer::Program* flatbuffer_program =
      ProgramTestFriend::GetInternalProgram(&program.get());

  // The constant buffer should be empty.
  EXPECT_EQ(flatbuffer_program->constant_buffer()->size(), 0);

  // Expect 1 constant segment, placeholder for non-const tensors.
  EXPECT_EQ(flatbuffer_program->segments()->size(), 1);
}

TEST_F(ProgramTest, LoadConstantSegment) {
  // Load the serialized ModuleAddMul data, with constants in the segment.
  const char* linear_path = std::getenv("ET_MODULE_ADD_MUL_PATH");
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

TEST_F(ProgramTest, LoadConstantSegmentWhenConstantBufferExists) {
  // Load the serialized ModuleAddMul data, with constants in the flatbuffer and
  // no constants in the segment.
  const char* linear_path =
      std::getenv("DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH");
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
#if ET_ENABLE_DEPRECATED_CONSTANT_BUFFER
  ASSERT_EQ(program.error(), Error::Ok);

  const executorch_flatbuffer::Program* flatbuffer_program =
      ProgramTestFriend::GetInternalProgram(&program.get());

  // Expect no segments.
  EXPECT_EQ(flatbuffer_program->segments()->size(), 0);

  // The constant buffer should exist.
  EXPECT_GE(flatbuffer_program->constant_buffer()->size(), 1);
#else
  // The deprecated in-flatbuffer constant_buffer path is disabled; loading a
  // PTE that relies on it must be rejected.
  EXPECT_EQ(program.error(), Error::InvalidProgram);
#endif
}

TEST_F(ProgramTest, LoadFromMutableSegment) {
  // Load the serialized ModuleSimpleTrain data.
  auto path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
  Result<FileDataLoader> training_loader = FileDataLoader::from(path);
  ASSERT_EQ(training_loader.error(), Error::Ok);

  // This file should always be compatible.
  Result<FreeableBuffer> training_header = training_loader->load(
      /*offset=*/0,
      Program::kMinHeadBytes,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(training_header.error(), Error::Ok);
  EXPECT_EQ(
      Program::check_header(training_header->data(), training_header->size()),
      Program::HeaderStatus::CompatibleVersion);

  Result<Program> program = Program::load(&training_loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  // The linear layer weight tensor is 3x3 floats = 36 bytes.
  // Load the full weight data to verify the loading mechanism works correctly.
  constexpr size_t kWeightSize = 36;
  uint8_t buffer[kWeightSize] = {0};
  uint8_t buffer2[kWeightSize] = {0};

  // Load the mutable segment data (linear.weight at offset_index 1)
  Error err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 0, 1, kWeightSize, buffer);
  EXPECT_EQ(err, Error::Ok);

  // Load the same data into a second buffer
  err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 0, 1, kWeightSize, buffer2);
  EXPECT_EQ(err, Error::Ok);

  // Verify both loads return identical data
  EXPECT_EQ(std::memcmp(buffer, buffer2, kWeightSize), 0)
      << "Loading the same segment twice should return identical data";

  // Loading into a new buffer, as for device memory, returns the same data.
  Result<FreeableBuffer> loaded = ProgramTestFriend::load_mutable_subsegment(
      &program.get(), 0, 1, kWeightSize);
  ASSERT_EQ(loaded.error(), Error::Ok);
  ASSERT_EQ(loaded->size(), kWeightSize);
  EXPECT_EQ(std::memcmp(loaded->data(), buffer, kWeightSize), 0);

  // Verify that the loaded data is non-trivial (not all zeros).
  bool has_nonzero = false;
  for (size_t i = 0; i < kWeightSize; ++i) {
    if (buffer[i] != 0) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero)
      << "Loaded mutable segment data should not be all zeros";

  // Mutate the local buffer and reload to verify source immutability
  std::memset(buffer, 0xFF, kWeightSize);
  err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 0, 1, kWeightSize, buffer);
  EXPECT_EQ(err, Error::Ok);

  // Verify the reloaded data matches the original (local mutation didn't
  // affect the file source)
  EXPECT_EQ(std::memcmp(buffer, buffer2, kWeightSize), 0)
      << "Reloading after local mutation should return original data";

  const executorch_flatbuffer::Program* flatbuffer_program =
      ProgramTestFriend::GetInternalProgram(&program.get());

  // Expect 2 segments. 1 mutable segment and 1 constant segment.
  EXPECT_EQ(flatbuffer_program->segments()->size(), 2);

  // Expect a mutable data segment.
  EXPECT_EQ(flatbuffer_program->mutable_data_segments()->size(), 1);

  // Expect the 0 index to be reserved and the offsets for weight and bias of
  // linear to be indices 1 and 2.
  EXPECT_EQ(
      flatbuffer_program->mutable_data_segments()->Get(0)->offsets()->size(),
      3);
  EXPECT_EQ(
      flatbuffer_program->mutable_data_segments()->Get(0)->offsets()->Get(0),
      0);
  EXPECT_EQ(
      flatbuffer_program->mutable_data_segments()->Get(0)->offsets()->Get(1),
      0);
  EXPECT_EQ(
      flatbuffer_program->mutable_data_segments()->Get(0)->offsets()->Get(2),
      36);

  // Loading beyond file should fail
  err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 0, 1, 500, buffer);
  EXPECT_NE(err, Error::Ok);

  // Loading beyond offsets should fail
  err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 0, 500, 1, buffer);
  EXPECT_NE(err, Error::Ok);

  // Loading beyond segments should fail
  err = ProgramTestFriend::load_mutable_subsegment_into(
      &program.get(), 500, 1, 1, buffer);
  EXPECT_NE(err, Error::Ok);

  // The new buffer variant checks the same bounds.
  EXPECT_NE(
      ProgramTestFriend::load_mutable_subsegment(&program.get(), 0, 1, 500)
          .error(),
      Error::Ok);
  EXPECT_NE(
      ProgramTestFriend::load_mutable_subsegment(&program.get(), 0, 500, 1)
          .error(),
      Error::Ok);
}

TEST_F(ProgramTest, LoadAndCheckPTESize) {
  // Load the serialized ModuleAddMul data, with constants in the segment.
  const char* linear_path = std::getenv("ET_MODULE_ADD_MUL_PATH");
  Result<FileDataLoader> linear_loader = FileDataLoader::from(linear_path);
  ASSERT_EQ(linear_loader.error(), Error::Ok);
  Result<Program> program = Program::load(&linear_loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  // Create a truncated file.
  Result<FreeableBuffer> truncated_file = linear_loader->load(
      0, 200, DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(truncated_file.error(), Error::Ok);

  BufferDataLoader truncated_loader =
      BufferDataLoader(truncated_file->data(), 200);
  Result<Program> truncated_program = Program::load(&truncated_loader);
  ASSERT_EQ(truncated_program.error(), Error::InvalidProgram);
}

// Regression test for null pointer dereference in
// get_output_flattening_encoding when container_meta_type is missing from the
// ExecutionPlan.
TEST_F(ProgramTest, GetOutputFlatteningEncodingWithMissingContainerMetaType) {
  // Build a minimal valid Program with an ExecutionPlan that has NO
  // container_meta_type. This would previously crash with a null pointer
  // dereference.
  flatbuffers::FlatBufferBuilder builder(1024);

  // Create a minimal ExecutionPlan with name but NO container_meta_type
  auto plan_name = builder.CreateString("forward");

  // Create empty vectors for required fields
  auto empty_values = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{});
  auto empty_inputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_outputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  auto empty_operators = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  auto empty_delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  auto buffer_sizes = builder.CreateVector(std::vector<int64_t>{0});

  // Build ExecutionPlan WITHOUT container_meta_type (it will be null)
  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      0, // container_meta_type = null (this is the vulnerability trigger)
      empty_values,
      empty_inputs,
      empty_outputs,
      empty_chains,
      empty_operators,
      empty_delegates,
      buffer_sizes);

  auto execution_plans = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
          execution_plan});

  // Create empty segment info
  auto empty_constant_buffer = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Buffer>>{});
  auto empty_backend_data = builder.CreateVector(
      std::vector<flatbuffers::Offset<
          executorch_flatbuffer::BackendDelegateInlineData>>{});
  auto empty_segments = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{});

  // Placeholder constant_segment.
  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));

  // Build the Program
  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      0, // version
      execution_plans,
      empty_constant_buffer,
      empty_backend_data,
      empty_segments,
      constant_segment);

  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  // Copy to 16-byte aligned buffer (required by Program::load)
  alignas(16) uint8_t aligned_buffer[2048];
  ASSERT_LE(builder.GetSize(), sizeof(aligned_buffer));
  std::memcpy(aligned_buffer, builder.GetBufferPointer(), builder.GetSize());

  // Load the malformed program
  BufferDataLoader data_loader(aligned_buffer, builder.GetSize());
  Result<Program> loaded_program =
      Program::load(&data_loader, Program::Verification::Minimal);

  // Program should load successfully (the malformed data is valid FlatBuffer)
  ASSERT_EQ(loaded_program.error(), Error::Ok);

  // Calling get_output_flattening_encoding should return an error,
  // NOT crash with null pointer dereference
  Result<const char*> encoding =
      loaded_program->get_output_flattening_encoding("forward");

  // Should return InvalidProgram error due to missing container_meta_type
  EXPECT_EQ(encoding.error(), Error::InvalidProgram);
}

// Test that get_output_flattening_encoding handles missing encoded_out_str
TEST_F(ProgramTest, GetOutputFlatteningEncodingWithMissingEncodedOutStr) {
  flatbuffers::FlatBufferBuilder builder(2048);

  // Create ContainerMetadata with encoded_inp_str but NO encoded_out_str
  auto inp_str = builder.CreateString("test_input");
  auto container_meta = executorch_flatbuffer::CreateContainerMetadata(
      builder,
      inp_str,
      0); // encoded_out_str = null

  auto plan_name = builder.CreateString("forward");
  auto empty_values = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{});
  auto empty_inputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_outputs = builder.CreateVector(std::vector<int32_t>{});
  auto empty_chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{});
  auto empty_operators = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{});
  auto empty_delegates = builder.CreateVector(
      std::vector<
          flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{});
  auto buffer_sizes = builder.CreateVector(std::vector<int64_t>{0});

  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      plan_name,
      container_meta, // container_meta_type exists
      empty_values,
      empty_inputs,
      empty_outputs,
      empty_chains,
      empty_operators,
      empty_delegates,
      buffer_sizes);

  auto execution_plans = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
          execution_plan});

  auto empty_constant_buffer = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Buffer>>{});
  auto empty_backend_data = builder.CreateVector(
      std::vector<flatbuffers::Offset<
          executorch_flatbuffer::BackendDelegateInlineData>>{});
  auto empty_segments = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{});

  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      0,
      execution_plans,
      empty_constant_buffer,
      empty_backend_data,
      empty_segments,
      constant_segment);

  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  // Copy to 16-byte aligned buffer (required by Program::load)
  alignas(16) uint8_t aligned_buffer[2048];
  ASSERT_LE(builder.GetSize(), sizeof(aligned_buffer));
  std::memcpy(aligned_buffer, builder.GetBufferPointer(), builder.GetSize());

  BufferDataLoader data_loader(aligned_buffer, builder.GetSize());
  Result<Program> loaded_program =
      Program::load(&data_loader, Program::Verification::Minimal);

  ASSERT_EQ(loaded_program.error(), Error::Ok);

  // Should return error for missing encoded_out_str, not crash
  Result<const char*> encoding =
      loaded_program->get_output_flattening_encoding("forward");

  EXPECT_EQ(encoding.error(), Error::InvalidProgram);
}

TEST_F(ProgramTest, NullPlanNameDoesNotCrash) {
  flatbuffers::FlatBufferBuilder builder(1024);

  // Create dummy execution_plan with null plan_name.
  auto execution_plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      0,
      0, // name=null, container_meta_type=null
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>{}),
      builder.CreateVector(std::vector<int32_t>{}),
      builder.CreateVector(std::vector<int32_t>{}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{}),
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>{}),
      builder.CreateVector(std::vector<int64_t>{0}));

  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      0,
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{
              execution_plan}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::Buffer>>{}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<
              executorch_flatbuffer::BackendDelegateInlineData>>{}),
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{}),
      constant_segment);

  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());

  alignas(16) uint8_t buf[2048];
  std::memcpy(buf, builder.GetBufferPointer(), builder.GetSize());
  BufferDataLoader loader(buf, builder.GetSize());

  Result<Program> p = Program::load(&loader, Program::Verification::Minimal);
  ASSERT_EQ(p.error(), Error::Ok);

  // Should return error, not crash
  EXPECT_EQ(p->method_meta("forward").error(), Error::InvalidArgument);
}

#if ET_ENABLE_DEPRECATED_CONSTANT_BUFFER
TEST_F(ProgramTest, GetConstantBufferDataRejectsOversizedRequest) {
  const char* path =
      std::getenv("DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH");
  Result<FileDataLoader> loader = FileDataLoader::from(path);
  ASSERT_EQ(loader.error(), Error::Ok);

  Result<Program> program = Program::load(&loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  // Request way more bytes than any buffer could contain
  Result<const void*> data = ProgramTestFriend::get_constant_buffer_data(
      &program.get(), /*buffer_index=*/0, /*nbytes=*/SIZE_MAX);

  EXPECT_EQ(data.error(), Error::InvalidArgument);
}
#endif // ET_ENABLE_DEPRECATED_CONSTANT_BUFFER

namespace {

constexpr const char* kCompileSpecTestBackendId = "CompileSpecTestBackend";

class CompileSpecTestBackend final
    : public executorch::runtime::BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext&,
      FreeableBuffer*,
      executorch::aten::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override {
    for (const auto& spec : compile_specs) {
      if (spec.key == nullptr) {
        return Error::InvalidProgram;
      }
      if (spec.value.nbytes > 0 && spec.value.buffer == nullptr) {
        return Error::InvalidProgram;
      }
    }
    return reinterpret_cast<executorch::runtime::DelegateHandle*>(
        &handle_token_);
  }

  Error execute(
      executorch::runtime::BackendExecutionContext&,
      executorch::runtime::DelegateHandle*,
      executorch::runtime::Span<executorch::runtime::EValue*>) const override {
    return Error::Ok;
  }

 private:
  mutable int handle_token_ = 1;
};

Error ensure_compile_spec_test_backend_registered() {
  static CompileSpecTestBackend backend;
  if (executorch::runtime::get_backend_class(kCompileSpecTestBackendId) ==
      nullptr) {
    return executorch::runtime::register_backend(
        {kCompileSpecTestBackendId, &backend});
  }
  return Error::Ok;
}

enum class TestEValueKind {
  Null,
  Int,
  Tensor,
  TensorList,
  OptionalTensorList,
};

struct TestEValueSpec {
  TestEValueKind kind;
  std::vector<int32_t> items = {};
  bool has_items = true;
};

struct TestCompileSpecEntry {
  bool has_key = true;
  std::string key = "";
  bool has_value = true;
  std::vector<uint8_t> value = {};
};

class AlignedProgramBuffer final {
 public:
  explicit AlignedProgramBuffer(const flatbuffers::FlatBufferBuilder& builder)
      : storage_(builder.GetSize() + 16), size_(builder.GetSize()) {
    uintptr_t raw = reinterpret_cast<uintptr_t>(storage_.data());
    uintptr_t aligned = (raw + uintptr_t{15}) & ~uintptr_t{15};
    aligned_ptr_ = reinterpret_cast<uint8_t*>(aligned);
    std::memcpy(aligned_ptr_, builder.GetBufferPointer(), size_);
  }

  BufferDataLoader loader() const {
    return BufferDataLoader(aligned_ptr_, size_);
  }

 private:
  std::vector<uint8_t> storage_;
  uint8_t* aligned_ptr_ = nullptr;
  size_t size_ = 0;
};

AlignedProgramBuffer build_list_and_delegate_test_program(
    const std::vector<TestEValueSpec>& values_spec,
    const std::vector<TestCompileSpecEntry>* delegate_specs = nullptr) {
  flatbuffers::FlatBufferBuilder builder(2048);

  std::vector<flatbuffers::Offset<executorch_flatbuffer::EValue>> evalues;
  evalues.reserve(values_spec.size());
  for (const auto& spec : values_spec) {
    switch (spec.kind) {
      case TestEValueKind::Null: {
        auto null_val = executorch_flatbuffer::CreateNull(builder);
        evalues.push_back(
            executorch_flatbuffer::CreateEValue(
                builder,
                executorch_flatbuffer::KernelTypes::Null,
                null_val.Union()));
        break;
      }
      case TestEValueKind::Int: {
        auto int_val = executorch_flatbuffer::CreateInt(builder, 99);
        evalues.push_back(
            executorch_flatbuffer::CreateEValue(
                builder,
                executorch_flatbuffer::KernelTypes::Int,
                int_val.Union()));
        break;
      }
      case TestEValueKind::Tensor: {
        auto sizes = builder.CreateVector(std::vector<int32_t>{2});
        auto dim_order = builder.CreateVector(std::vector<uint8_t>{0});
        auto tensor = executorch_flatbuffer::CreateTensor(
            builder,
            executorch_flatbuffer::ScalarType::FLOAT,
            /*storage_offset=*/0,
            sizes,
            dim_order,
            /*requires_grad=*/false,
            /*data_buffer_idx=*/0,
            /*allocation_info=*/0,
            /*layout=*/0,
            executorch_flatbuffer::TensorShapeDynamism::STATIC,
            /*extra_tensor_info=*/0);
        evalues.push_back(
            executorch_flatbuffer::CreateEValue(
                builder,
                executorch_flatbuffer::KernelTypes::Tensor,
                tensor.Union()));
        break;
      }
      case TestEValueKind::TensorList: {
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> items_off =
            spec.has_items ? builder.CreateVector(spec.items) : 0;
        auto list_val =
            executorch_flatbuffer::CreateTensorList(builder, items_off);
        evalues.push_back(
            executorch_flatbuffer::CreateEValue(
                builder,
                executorch_flatbuffer::KernelTypes::TensorList,
                list_val.Union()));
        break;
      }
      case TestEValueKind::OptionalTensorList: {
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> items_off =
            spec.has_items ? builder.CreateVector(spec.items) : 0;
        auto list_val =
            executorch_flatbuffer::CreateOptionalTensorList(builder, items_off);
        evalues.push_back(
            executorch_flatbuffer::CreateEValue(
                builder,
                executorch_flatbuffer::KernelTypes::OptionalTensorList,
                list_val.Union()));
        break;
      }
    }
  }

  auto empty_instructions = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Instruction>>{});
  auto chain = executorch_flatbuffer::CreateChain(
      builder,
      builder.CreateVector(std::vector<int32_t>{}),
      builder.CreateVector(std::vector<int32_t>{}),
      empty_instructions);
  auto chains = builder.CreateVector(
      std::vector<flatbuffers::Offset<executorch_flatbuffer::Chain>>{chain});

  std::vector<flatbuffers::Offset<executorch_flatbuffer::BackendDelegate>>
      delegates_vec;
  std::vector<
      flatbuffers::Offset<executorch_flatbuffer::BackendDelegateInlineData>>
      inline_data_vec;
  if (delegate_specs != nullptr) {
    std::vector<flatbuffers::Offset<executorch_flatbuffer::CompileSpec>>
        fb_specs;
    fb_specs.reserve(delegate_specs->size());
    for (const auto& cs : *delegate_specs) {
      flatbuffers::Offset<flatbuffers::String> key_off =
          cs.has_key ? builder.CreateString(cs.key) : 0;
      flatbuffers::Offset<flatbuffers::Vector<uint8_t>> val_off =
          cs.has_value ? builder.CreateVector(cs.value) : 0;
      fb_specs.push_back(
          executorch_flatbuffer::CreateCompileSpec(builder, key_off, val_off));
    }
    auto data_ref = executorch_flatbuffer::CreateBackendDelegateDataReference(
        builder, executorch_flatbuffer::DataLocation::INLINE, /*index=*/0);
    delegates_vec.push_back(
        executorch_flatbuffer::CreateBackendDelegate(
            builder,
            builder.CreateString(kCompileSpecTestBackendId),
            data_ref,
            builder.CreateVector(fb_specs)));
    inline_data_vec.push_back(
        executorch_flatbuffer::CreateBackendDelegateInlineData(
            builder, builder.CreateVector(std::vector<uint8_t>{0x01, 0x02})));
  }

  auto plan = executorch_flatbuffer::CreateExecutionPlan(
      builder,
      builder.CreateString("forward"),
      /*container_meta_type=*/0,
      builder.CreateVector(evalues),
      /*inputs=*/builder.CreateVector(std::vector<int32_t>{}),
      /*outputs=*/builder.CreateVector(std::vector<int32_t>{}),
      chains,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::Operator>>{}),
      builder.CreateVector(delegates_vec),
      /*non_const_buffer_sizes=*/
      builder.CreateVector(std::vector<int64_t>{}));

  auto constant_segment = executorch_flatbuffer::CreateSubsegmentOffsets(
      builder,
      /*segment_index=*/0,
      builder.CreateVector(std::vector<uint64_t>{0}));

  auto program = executorch_flatbuffer::CreateProgram(
      builder,
      /*version=*/0,
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::ExecutionPlan>>{plan}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<executorch_flatbuffer::Buffer>>{}),
      builder.CreateVector(inline_data_vec),
      builder.CreateVector(
          std::vector<
              flatbuffers::Offset<executorch_flatbuffer::DataSegment>>{}),
      constant_segment);

  builder.Finish(program, executorch_flatbuffer::ProgramIdentifier());
  return AlignedProgramBuffer(builder);
}

} // namespace

TEST_F(ProgramTest, ValidTopologicalListsAndCompileSpecsSucceed) {
  ASSERT_EQ(ensure_compile_spec_test_backend_registered(), Error::Ok);

  std::vector<TestCompileSpecEntry> specs = {
      {/*has_key=*/true, "precision", /*has_value=*/true, {1, 2, 3, 4}},
      {/*has_key=*/true, "", /*has_value=*/true, {}},
  };
  AlignedProgramBuffer buf = build_list_and_delegate_test_program(
      {
          {TestEValueKind::Tensor},
          {TestEValueKind::Null},
          {TestEValueKind::TensorList, {0}},
          {TestEValueKind::OptionalTensorList, {-1, 0, 1}},
      },
      &specs);
  BufferDataLoader loader = buf.loader();

  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  ASSERT_EQ(program.error(), Error::Ok);

  alignas(64) uint8_t method_buf[16 * 1024] = {};
  executorch::runtime::MemoryAllocator method_allocator(
      sizeof(method_buf), method_buf);
  executorch::runtime::MemoryManager memory_manager(&method_allocator);

  Result<executorch::runtime::Method> method =
      program->load_method("forward", &memory_manager);
  EXPECT_EQ(method.error(), Error::Ok);
}

#if ET_ENABLE_PROGRAM_VERIFICATION
TEST_F(ProgramTest, ValidateProgramRejectsForwardAndSelfReferencedTensorList) {
  {
    AlignedProgramBuffer forward_ref = build_list_and_delegate_test_program({
        {TestEValueKind::TensorList, {1}},
        {TestEValueKind::Tensor},
    });
    BufferDataLoader loader = forward_ref.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer self_ref = build_list_and_delegate_test_program({
        {TestEValueKind::TensorList, {0}},
    });
    BufferDataLoader loader = self_ref.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer partial_forward_ref =
        build_list_and_delegate_test_program({
            {TestEValueKind::Tensor},
            {TestEValueKind::TensorList, {0, 2}},
            {TestEValueKind::Tensor},
        });
    BufferDataLoader loader = partial_forward_ref.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
}

TEST_F(ProgramTest, ValidateProgramRejectsInvalidOptionalTensorListItems) {
  {
    AlignedProgramBuffer forward_ref = build_list_and_delegate_test_program({
        {TestEValueKind::OptionalTensorList, {1}},
        {TestEValueKind::Tensor},
    });
    BufferDataLoader loader = forward_ref.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer self_ref = build_list_and_delegate_test_program({
        {TestEValueKind::OptionalTensorList, {0}},
    });
    BufferDataLoader loader = self_ref.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer oob_pos = build_list_and_delegate_test_program({
        {TestEValueKind::Tensor},
        {TestEValueKind::OptionalTensorList, {-1, 99}},
    });
    BufferDataLoader loader = oob_pos.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer oob_neg = build_list_and_delegate_test_program({
        {TestEValueKind::Tensor},
        {TestEValueKind::OptionalTensorList, {-2}},
    });
    BufferDataLoader loader = oob_neg.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer wrong_type = build_list_and_delegate_test_program({
        {TestEValueKind::Int},
        {TestEValueKind::OptionalTensorList, {0}},
    });
    BufferDataLoader loader = wrong_type.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
  {
    AlignedProgramBuffer missing_items = build_list_and_delegate_test_program({
        {TestEValueKind::OptionalTensorList, {}, /*has_items=*/false},
    });
    BufferDataLoader loader = missing_items.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::InternalConsistency);
    EXPECT_EQ(program.error(), Error::InvalidProgram);
  }
}

#endif // ET_ENABLE_PROGRAM_VERIFICATION

TEST_F(
    ProgramTest,
    LoadMethodMinimalVerificationRejectsForwardReferencedTensorLists) {
  alignas(64) uint8_t reused_buf[16 * 1024] = {};
  executorch::runtime::MemoryAllocator reused_allocator(
      sizeof(reused_buf), reused_buf);
  executorch::runtime::MemoryManager reused_mm(&reused_allocator);

  // Seed the arena with a valid 2-Tensor Method, then destroy and reset it so
  // unconstructed EValue slots contain stale Tag::Tensor tags on reuse.
  {
    AlignedProgramBuffer seed = build_list_and_delegate_test_program({
        {TestEValueKind::Tensor},
        {TestEValueKind::Tensor},
    });
    BufferDataLoader seed_loader = seed.loader();
    Result<Program> seed_prog =
        Program::load(&seed_loader, Program::Verification::Minimal);
    ASSERT_EQ(seed_prog.error(), Error::Ok);
    Result<executorch::runtime::Method> seed_method =
        seed_prog->load_method("forward", &reused_mm);
    ASSERT_EQ(seed_method.error(), Error::Ok);
  }
  reused_allocator.reset();

  // Forward-referenced TensorList([1]) at index 0 must be rejected even when
  // Minimal verification skips validate_program() and values_[1] holds stale
  // Tag::Tensor bytes in the reused allocator arena.
  {
    AlignedProgramBuffer forward_tl = build_list_and_delegate_test_program({
        {TestEValueKind::TensorList, {1}},
        {TestEValueKind::Int},
    });
    BufferDataLoader loader = forward_tl.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &reused_mm);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }

  // Re-seed 1 Tensor slot and verify self-reference TensorList([0]) is
  // rejected.
  reused_allocator.reset();
  {
    AlignedProgramBuffer seed1 = build_list_and_delegate_test_program({
        {TestEValueKind::Tensor},
    });
    BufferDataLoader seed_loader = seed1.loader();
    Result<Program> seed_prog =
        Program::load(&seed_loader, Program::Verification::Minimal);
    ASSERT_EQ(seed_prog.error(), Error::Ok);
    Result<executorch::runtime::Method> seed_method =
        seed_prog->load_method("forward", &reused_mm);
    ASSERT_EQ(seed_method.error(), Error::Ok);
  }
  reused_allocator.reset();
  {
    AlignedProgramBuffer self_tl = build_list_and_delegate_test_program({
        {TestEValueKind::TensorList, {0}},
    });
    BufferDataLoader loader = self_tl.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &reused_mm);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }

  // Forward-referenced and self-referenced OptionalTensorList must also be
  // rejected on a zeroed arena (where unconstructed EValues look like
  // Tag::None == 0).
  alignas(64) uint8_t clean_buf[16 * 1024] = {};
  executorch::runtime::MemoryAllocator clean_allocator(
      sizeof(clean_buf), clean_buf);
  executorch::runtime::MemoryManager clean_mm(&clean_allocator);
  {
    AlignedProgramBuffer forward_otl = build_list_and_delegate_test_program({
        {TestEValueKind::OptionalTensorList, {1}},
        {TestEValueKind::Int},
    });
    BufferDataLoader loader = forward_otl.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &clean_mm);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }
  clean_allocator.reset();
  {
    AlignedProgramBuffer self_otl = build_list_and_delegate_test_program({
        {TestEValueKind::OptionalTensorList, {0}},
    });
    BufferDataLoader loader = self_otl.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &clean_mm);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }
}

TEST_F(ProgramTest, LoadMethodRejectsCompileSpecWithNullKeyOrValue) {
  ASSERT_EQ(ensure_compile_spec_test_backend_registered(), Error::Ok);

  {
    std::vector<TestCompileSpecEntry> null_key_specs = {
        {/*has_key=*/false, "", /*has_value=*/true, {1, 2}},
    };
    AlignedProgramBuffer buf =
        build_list_and_delegate_test_program({}, &null_key_specs);
    BufferDataLoader loader = buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);

    alignas(64) uint8_t method_buf[16 * 1024] = {};
    executorch::runtime::MemoryAllocator method_allocator(
        sizeof(method_buf), method_buf);
    executorch::runtime::MemoryManager memory_manager(&method_allocator);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &memory_manager);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }
  {
    std::vector<TestCompileSpecEntry> null_value_specs = {
        {/*has_key=*/true, "precision", /*has_value=*/false, {}},
    };
    AlignedProgramBuffer buf =
        build_list_and_delegate_test_program({}, &null_value_specs);
    BufferDataLoader loader = buf.loader();
    Result<Program> program =
        Program::load(&loader, Program::Verification::Minimal);
    ASSERT_EQ(program.error(), Error::Ok);

    alignas(64) uint8_t method_buf[16 * 1024] = {};
    executorch::runtime::MemoryAllocator method_allocator(
        sizeof(method_buf), method_buf);
    executorch::runtime::MemoryManager memory_manager(&method_allocator);
    Result<executorch::runtime::Method> method =
        program->load_method("forward", &memory_manager);
    EXPECT_EQ(method.error(), Error::InvalidProgram);
  }
}
