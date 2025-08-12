/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <memory>

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
using executorch::runtime::Span;

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
      __ET_UNUSED Span<EValue*> args) const override {
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
        if (std::holds_alternative<
                std::array<char, executorch::runtime::kMaxOptionValueLength>>(
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
  Span<EValue*> args((EValue**)nullptr, (size_t)0); // Not used in mock
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
  Span<EValue*> args((EValue**)nullptr, (size_t)0); // Not used in mock
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

// Mock backend for testing
class StubBackend : public BackendInterface {
 public:
  ~StubBackend() override = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    return nullptr;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    return Error::Ok;
  }

  Error get_option(
      BackendOptionContext& context,
      executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    // For testing purposes, just record that get_option was called
    // and verify the input parameters
    get_option_called = true;
    get_option_call_count++;
    last_get_option_size = backend_options.size();

    // Verify that the expected option key is present and modify the value
    for (size_t i = 0; i < backend_options.size(); ++i) {
      if (strcmp(backend_options[i].key, "NumberOfThreads") == 0) {
        // Set the value to what was stored by set_option
        backend_options[i].value = last_num_threads;
        found_expected_key = true;
        break;
      }
    }

    return Error::Ok;
  }

  Error set_option(
      BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override {
    // Store the options for verification
    last_options_size = backend_options.size();
    if (backend_options.size() > 0) {
      for (const auto& option : backend_options) {
        if (strcmp(option.key, "NumberOfThreads") == 0) {
          if (auto* val = std::get_if<int>(&option.value)) {
            last_num_threads = *val;
          }
        }
      }
    }
    return Error::Ok;
  }

  // Mutable for testing verification
  size_t last_options_size = 0;
  int last_num_threads = 0;
  bool get_option_called = false;
  int get_option_call_count = 0;
  size_t last_get_option_size = 0;
  bool found_expected_key = false;
};

class BackendUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Register the stub backend
    stub_backend = std::make_unique<StubBackend>();
    Backend backend_config{"StubBackend", stub_backend.get()};
    auto register_result = register_backend(backend_config);
    ASSERT_EQ(register_result, Error::Ok);
  }

  std::unique_ptr<StubBackend> stub_backend;
};

// Test basic string functionality
TEST_F(BackendUpdateTest, TestSetGetOption) {
  BackendOptions<1> backend_options;
  int new_num_threads = 4;
  backend_options.set_option("NumberOfThreads", new_num_threads);

  auto status = set_option("StubBackend", backend_options.view());
  ASSERT_EQ(status, Error::Ok);

  // Set up the default option, which will be populuated by the get_option call
  BackendOption ref_backend_option{"NumberOfThreads", 0};
  status = get_option("StubBackend", ref_backend_option);

  // Verify that the backend actually received the options
  ASSERT_TRUE(std::get<int>(ref_backend_option.value) == new_num_threads);

  // Verify that the backend actually update the options
  ASSERT_EQ(stub_backend->last_options_size, 1);
  ASSERT_EQ(stub_backend->last_num_threads, new_num_threads);
}
