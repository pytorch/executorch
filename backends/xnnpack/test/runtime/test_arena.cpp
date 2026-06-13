#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/executor/arena.h>

using namespace executorch::backends::xnnpack::executor;
using executorch::runtime::Error;

TEST(TestArena, initial_empty) {
  Arena arena;
  EXPECT_EQ(arena.size, 0u);
  EXPECT_EQ(arena.data(), nullptr);
}

TEST(TestArena, grow_allocates) {
  Arena arena;
  EXPECT_EQ(arena.resize(128), Error::Ok);
  EXPECT_EQ(arena.size, 128u);
  EXPECT_NE(arena.data(), nullptr);
}

TEST(TestArena, shrink_is_noop) {
  Arena arena;
  ASSERT_EQ(arena.resize(128), Error::Ok);
  void* data_before = arena.data();

  // A smaller (or equal) request neither reallocates nor shrinks.
  EXPECT_EQ(arena.resize(64), Error::Ok);
  EXPECT_EQ(arena.size, 128u);
  EXPECT_EQ(arena.data(), data_before);
}

TEST(TestArena, grow_again) {
  Arena arena;
  ASSERT_EQ(arena.resize(64), Error::Ok);
  EXPECT_EQ(arena.resize(256), Error::Ok);
  EXPECT_EQ(arena.size, 256u);
  EXPECT_NE(arena.data(), nullptr);
}
