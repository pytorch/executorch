/**
 * @file
 * Custom main function for unit test executables.
 * Based on `fbcode//common/gtest/LightMain.cpp`
 */

#include <executorch/compiler/Compiler.h>
#include <executorch/core/Runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

// Define main as a weak symbol to allow trivial overriding.
int main(int argc, char** argv) __ET_WEAK;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv);
  torch::executor::runtime_init();
  return RUN_ALL_TESTS();
}
