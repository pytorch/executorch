// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Program.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {
namespace {

std::vector<uint8_t> make_program(const std::string& version) {
  flatbuffers::FlatBufferBuilder builder;
  const auto serialized_version = builder.CreateString(version);
  const auto methods = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Method>>{});
  const auto program =
      native_backend::CreateProgram(builder, serialized_version, methods);
  native_backend::FinishProgramBuffer(builder, program);
  return {
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize()};
}

TEST(ProgramVersionTest, Load_CurrentVersion_ReturnsParsedVersion) {
  const auto bytes = make_program("1.0");

  const Program program = Program::load(bytes.data(), bytes.size());

  EXPECT_EQ(program.version().major, 1);
  EXPECT_EQ(program.version().minor, 0);
}

TEST(ProgramVersionTest, MoveConstructionPreservesVersion) {
  const auto bytes = make_program("1.0");
  Program source = Program::load(bytes.data(), bytes.size());

  const Program program(std::move(source));

  EXPECT_EQ(program.version().major, 1);
  EXPECT_EQ(program.version().minor, 0);
}

TEST(ProgramVersionTest, MoveAssignmentPreservesVersion) {
  const auto current_bytes = make_program("1.0");
  const auto legacy_bytes = make_program("1");
  Program source = Program::load(current_bytes.data(), current_bytes.size());
  Program destination = Program::load(legacy_bytes.data(), legacy_bytes.size());

  destination = std::move(source);

  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(source.flatbuffer(), nullptr);
  EXPECT_EQ(destination.version().major, 1);
  EXPECT_EQ(destination.version().minor, 0);
}

TEST(ProgramVersionTest, Load_LegacyMajorOnlyVersion_ReturnsVersionOne) {
  const auto bytes = make_program("1");

  const Program program = Program::load(bytes.data(), bytes.size());

  EXPECT_EQ(program.version().major, 1);
  EXPECT_EQ(program.version().minor, 0);
}

TEST(ProgramVersionTest, Load_FutureMinor_ThrowsUnsupportedVersion) {
  const auto bytes = make_program("1.1");

  EXPECT_THROW(
      Program::load(bytes.data(), bytes.size()), UnsupportedVersionError);
}

TEST(ProgramVersionTest, Load_DifferentMajor_ThrowsUnsupportedVersion) {
  const auto bytes = make_program("2.0");

  EXPECT_THROW(
      Program::load(bytes.data(), bytes.size()), UnsupportedVersionError);
}

TEST(ProgramVersionTest, Load_MalformedVersion_Throws) {
  const auto bytes = make_program("1.0.0");

  try {
    static_cast<void>(Program::load(bytes.data(), bytes.size()));
    FAIL() << "expected malformed version to be rejected";
  } catch (const UnsupportedVersionError&) {
    FAIL() << "malformed version was classified as unsupported";
  } catch (const std::runtime_error&) {
  }
}

TEST(ProgramVersionTest, Load_EmptyVersionComponent_Throws) {
  for (const std::string& version : {"", ".0", "1."}) {
    const auto bytes = make_program(version);
    EXPECT_THROW(Program::load(bytes.data(), bytes.size()), std::runtime_error)
        << version;
  }
}

TEST(ProgramVersionTest, Load_NonnumericVersion_Throws) {
  const auto bytes = make_program("1.x");

  EXPECT_THROW(Program::load(bytes.data(), bytes.size()), std::runtime_error);
}

TEST(ProgramVersionTest, Load_OverflowingVersion_Throws) {
  const auto bytes = make_program("4294967296.0");

  EXPECT_THROW(Program::load(bytes.data(), bytes.size()), std::runtime_error);
}

} // namespace
} // namespace ptn
