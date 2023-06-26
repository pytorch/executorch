#include <gtest/gtest.h>

#include <executorch/core/Log.h>
#include <executorch/core/Runtime.h>

namespace torch {
namespace executor {

class LoggingTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    // Initialize runtime.
    runtime_init();
  }
};

TEST_F(LoggingTest, LogLevels) {
  ET_LOG(Debug, "Debug log.");
  ET_LOG(Info, "Info log.");
  ET_LOG(Error, "Error log.");
  ET_LOG(Fatal, "Fatal log.");
}

TEST_F(LoggingTest, LogFormatting) {
  ET_LOG(Info, "Sample log with integer: %u", 100);
}

} // namespace executor
} // namespace torch
