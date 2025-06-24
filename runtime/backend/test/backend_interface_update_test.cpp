/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptionContext;
using executorch::runtime::BackendOptions;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::get_backend_class;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

class MockBackend : public BackendInterface {
 public:
  ~MockBackend() override = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      __ET_UNUSED BackendInitContext& context,
      __ET_UNUSED FreeableBuffer* processed,
      __ET_UNUSED ArrayRef<CompileSpec> compile_specs) const override {
    init_called = true;
    return nullptr;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      __ET_UNUSED DelegateHandle* handle,
      __ET_UNUSED EValue** args) const override {
    execute_count++;
    return Error::Ok;
  }

  Error set_option(
      __ET_UNUSED BackendOptionContext& context,
      const executorch::runtime::Span<BackendOption>& backend_options)
      override {
    set_option_count++;
    int success_update = 0;
    for (const auto& backend_option : backend_options) {
      if (strcmp(backend_option.key, "Backend") == 0) {
        if (std::holds_alternative<std::array<char, 256>>(
                backend_option.value)) {
          // Store the value in our member variable
          const auto& arr =
              std::get<std::array<char, 256>>(backend_option.value);
          target_backend = std::string(arr.data());
          success_update++;
        }
      } else if (strcmp(backend_option.key, "NumberOfThreads") == 0) {
        if (std::holds_alternative<int>(backend_option.value)) {
          num_threads = std::get<int>(backend_option.value);
          success_update++;
        }
      } else if (strcmp(backend_option.key, "Debug") == 0) {
        if (std::holds_alternative<bool>(backend_option.value)) {
          debug = std::get<bool>(backend_option.value);
          success_update++;
        }
      }
    }
    if (success_update == backend_options.size()) {
      return Error::Ok;
    }
    return Error::InvalidArgument;
  }

  // Mutable allows modification in const methods
  mutable std::optional<std::string> target_backend;
  mutable int num_threads = 0;
  mutable bool debug = false;

  // State tracking
  mutable bool init_called = false;
  mutable int execute_count = 0;
  mutable int set_option_count = 0;
};

class BackendInterfaceUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
    mock_backend = std::make_unique<MockBackend>();
    //  static Error register_success = register_executor_backend();
  }

  std::unique_ptr<MockBackend> mock_backend;
  BackendOptions<5> options;
};

TEST_F(BackendInterfaceUpdateTest, HandlesInvalidOption) {
  BackendOptionContext context;

  // Test invalid key case
  std::array<char, 256> value_array{"None"};
  BackendOption invalid_option{"InvalidKey", value_array};

  Error err = mock_backend->set_option(context, invalid_option);
  EXPECT_EQ(err, Error::InvalidArgument);
}

TEST_F(BackendInterfaceUpdateTest, HandlesStringOption) {
  BackendOptionContext context;
  options.set_option("Backend", "GPU");
  // // Create a backend option to pass to update

  EXPECT_EQ(mock_backend->target_backend, std::nullopt);

  // Test successful update
  Error err = mock_backend->set_option(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->target_backend, "GPU");
}

TEST_F(BackendInterfaceUpdateTest, HandlesIntOption) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendOptionContext context;

  int expected_num_threads = 4;

  // Create a backend option to pass to update
  options.set_option("NumberOfThreads", expected_num_threads);

  // Test successful update
  Error err = mock_backend->set_option(context, options.view());
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(mock_backend->num_threads, expected_num_threads);
}

TEST_F(BackendInterfaceUpdateTest, HandlesBoolOption) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendOptionContext context;

  options.set_option("Debug", true);

  // Test successful update
  Error err = mock_backend->set_option(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->debug, true);
}

TEST_F(BackendInterfaceUpdateTest, HandlesMultipleOptions) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendOptionContext context;

  options.set_option("Debug", true);
  options.set_option("NumberOfThreads", 4);
  options.set_option("Backend", "GPU");

  // Test successful update
  Error err = mock_backend->set_option(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->debug, true);
  EXPECT_EQ(mock_backend->num_threads, 4);
  EXPECT_EQ(mock_backend->target_backend, "GPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateBeforeInit) {
  BackendOptionContext option_context;
  MemoryAllocator memory_allocator{MemoryAllocator(0, nullptr)};

  BackendInitContext init_context(&memory_allocator);

  // Create backend option
  options.set_option("Backend", "GPU");

  // Update before init
  Error err = mock_backend->set_option(option_context, options.view());
  EXPECT_EQ(err, Error::Ok);

  // Now call init
  FreeableBuffer* processed = nullptr; // Not used in mock
  ArrayRef<CompileSpec> compile_specs; // Empty
  auto handle_or_error =
      mock_backend->init(init_context, processed, compile_specs);
  EXPECT_EQ(handle_or_error.error(), Error::Ok);

  // Verify state
  EXPECT_TRUE(mock_backend->init_called);
  EXPECT_EQ(mock_backend->set_option_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 0);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "GPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateAfterInitBeforeExecute) {
  BackendOptionContext option_context;
  MemoryAllocator init_memory_allocator{MemoryAllocator(0, nullptr)};
  BackendInitContext init_context(&init_memory_allocator);
  BackendExecutionContext execute_context;

  // First call init
  FreeableBuffer* processed = nullptr;
  ArrayRef<CompileSpec> compile_specs;
  auto handle_or_error =
      mock_backend->init(init_context, processed, compile_specs);
  EXPECT_TRUE(handle_or_error.ok());

  // Verify init called but execute not called
  EXPECT_TRUE(mock_backend->init_called);
  EXPECT_EQ(mock_backend->execute_count, 0);

  // Now update
  options.set_option("Backend", "CPU");
  Error err = mock_backend->set_option(option_context, options.view());
  EXPECT_EQ(err, Error::Ok);

  // Now execute
  DelegateHandle* handle = handle_or_error.get();
  EValue** args = nullptr; // Not used in mock
  err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);

  // Verify state
  EXPECT_EQ(mock_backend->set_option_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 1);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "CPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateBetweenExecutes) {
  BackendOptionContext option_context;
  MemoryAllocator init_memory_allocator{MemoryAllocator(0, nullptr)};
  BackendInitContext init_context(&init_memory_allocator);
  BackendExecutionContext execute_context;

  // Initialize
  FreeableBuffer* processed = nullptr;
  ArrayRef<CompileSpec> compile_specs;
  auto handle_or_error =
      mock_backend->init(init_context, processed, compile_specs);
  EXPECT_TRUE(handle_or_error.ok());
  DelegateHandle* handle = handle_or_error.get();

  // First execute
  EValue** args = nullptr;
  Error err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);

  // Update between executes
  options.set_option("Backend", "NPU");
  err = mock_backend->set_option(option_context, options.view());
  EXPECT_EQ(err, Error::Ok);

  // Second execute
  err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);

  // Verify state
  EXPECT_EQ(mock_backend->set_option_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 2);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "NPU");
}
