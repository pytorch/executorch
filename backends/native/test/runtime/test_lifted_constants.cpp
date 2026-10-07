// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/api/Model.h>

#include <array>
#include <cstdlib>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/Validation.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <gtest/gtest.h>

namespace ptn {
namespace {

// cppcheck-suppress-begin syntaxError
TEST(LiftedConstantsTest, LoadPreservesBindingsAndDeduplicatesStorage) {
  const char* path = std::getenv("ET_LIFTED_CONSTANTS_PATH");
  ASSERT_NE(path, nullptr);

  const Model model = Model::load_file(path);
  ASSERT_EQ(model.method_names().size(), 1);
  EXPECT_EQ(model.method_names()[0], "forward");

  const Package package = Package::load(path);
  const ByteSpan bytes = package.program_bytes();
  const Program program = Program::load(bytes.data(), bytes.size());
  const Method& method = program.get_method("forward");
  EXPECT_NO_THROW(validate_method_constants(method, package));
  ASSERT_EQ(method.data_bindings.size(), 2);
  EXPECT_NE(method.data_bindings[0].key, method.data_bindings[1].key);
  EXPECT_EQ(package.keys().size(), 2);
  EXPECT_EQ(package.owner_keys().size(), 1);
  EXPECT_EQ(package.aliases().size(), 1);
  EXPECT_EQ(package.constant_bytes(), 4 * sizeof(float));

  const std::array<float, 4> expected = {0.0f, 1.0f, 2.0f, 3.0f};
  for (const DataBinding& binding : method.data_bindings) {
    std::array<float, 4> values{};
    ASSERT_TRUE(package.load_constant_into(
        binding.key,
        MutableByteSpan(
            reinterpret_cast<uint8_t*>(values.data()), sizeof(values))));
    EXPECT_EQ(values, expected);
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
