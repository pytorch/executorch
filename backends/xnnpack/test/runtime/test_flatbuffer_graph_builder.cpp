#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/FlatbufferGraphBuilder.h>

#include <cstdint>
#include <vector>

using executorch::backends::xnnpack::FlatbufferGraphBuilder;

// Constructing a valid serialized graph requires the AOT serializer, so these
// cover the deserializer's rejection of malformed input: build() must return an
// error rather than aborting or reading out of bounds.

TEST(TestFlatbufferGraphBuilder, rejects_empty_buffer) {
  uint8_t dummy = 0;
  auto result = FlatbufferGraphBuilder::build(&dummy, 0);
  EXPECT_FALSE(result.ok());
}

TEST(TestFlatbufferGraphBuilder, rejects_garbage_buffer) {
  // No XNNHeader magic -> header parse fails -> error.
  std::vector<uint8_t> garbage(128, 0xAB);
  auto result = FlatbufferGraphBuilder::build(garbage.data(), garbage.size());
  EXPECT_FALSE(result.ok());
}
