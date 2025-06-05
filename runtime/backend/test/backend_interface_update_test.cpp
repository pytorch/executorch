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
using executorch::runtime::BackendInterface;
using executorch::runtime::Result;
using executorch::runtime::DelegateHandle;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::EValue;
using executorch::runtime::BackendUpdateContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Backend;
using executorch::runtime::StrKey;
using executorch::runtime::IntKey;
using executorch::runtime::BoolKey;
using executorch::runtime::get_backend_class;
using executorch::runtime::OptionType;
using executorch::runtime::MemoryAllocator;

class MockBackend : public BackendInterface {
  public:
      ~MockBackend() override = default;
  
      bool is_available() const override { return true; }
  
      Result<DelegateHandle*> init(
          BackendInitContext& context,
          FreeableBuffer* processed,
          ArrayRef<CompileSpec> compile_specs) const override {
            init_called = true;
            return nullptr;
      }
  
      Error execute(
          BackendExecutionContext& context,
          DelegateHandle* handle,
          EValue** args) const override {
            execute_count++;
            return Error::Ok;
      }
  
      Error update(
          BackendUpdateContext& context,
          const executorch::runtime::ArrayRef<BackendOption>& backend_options) const override {
          update_count++;
          int sucess_update = 0;
          for (const auto& backend_option : backend_options) {
            if (strcmp(backend_option.key, "Backend") == 0) {
                if (backend_option.type == OptionType::STRING) {
                    // Store the value in our member variable
                    target_backend = backend_option.value.string_value;
                    sucess_update++;
                }
            } else if (strcmp(backend_option.key, "NumberOfThreads") == 0) {
                if (backend_option.type == OptionType::INT) {
                  num_threads = backend_option.value.int_value;
                  sucess_update++;
                }
            } else if (strcmp(backend_option.key, "Debug") == 0) {
              if (backend_option.type == OptionType::BOOL) {
                debug = backend_option.value.bool_value;
                sucess_update++;
              }
            }
          }
          if (sucess_update == backend_options.size()) {
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
      mutable int update_count = 0;
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
  BackendUpdateContext context;
  
  // Test invalid key case
  BackendOption invalid_option{
    "InvalidKey",
    OptionType::STRING,
    {.string_value = "None"}
  };

  Error err = mock_backend->update(context, invalid_option);
  EXPECT_EQ(err, Error::InvalidArgument);

}

 TEST_F(BackendInterfaceUpdateTest, HandlesStringOption) {
  BackendUpdateContext context;
  options.set_option(StrKey("Backend"), "GPU");  
  // // Create a backend option to pass to update

  EXPECT_EQ(mock_backend->target_backend, std::nullopt);

  // Test successful update
  Error err = mock_backend->update(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->target_backend, "GPU");
}

TEST_F(BackendInterfaceUpdateTest, HandlesIntOption) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendUpdateContext context;

  int expected_num_threads = 4;
  
  // Create a backend option to pass to update
  options.set_option(IntKey("NumberOfThreads"), expected_num_threads);  
  
  // Test successful update
  Error err = mock_backend->update(context, options.view());
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(mock_backend->num_threads, expected_num_threads);
}

TEST_F(BackendInterfaceUpdateTest, HandlesBoolOption) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendUpdateContext context;
  
  options.set_option(BoolKey("Debug"), true);  
  
  // Test successful update
  Error err = mock_backend->update(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->debug, true);
}

TEST_F(BackendInterfaceUpdateTest, HandlesMultipleOptions) {
  // Check the default num_threads value is 0
  EXPECT_EQ(mock_backend->debug, false);
  // Create a mock context (needs to be defined or mocked)
  BackendUpdateContext context;
  
  options.set_option(BoolKey("Debug"), true);
  options.set_option(IntKey("NumberOfThreads"), 4);  
  options.set_option(StrKey("Backend"), "GPU");  
  
  // Test successful update
  Error err = mock_backend->update(context, options.view());
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(mock_backend->debug, true);
  EXPECT_EQ(mock_backend->num_threads, 4);
  EXPECT_EQ(mock_backend->target_backend, "GPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateBeforeInit) {
  BackendUpdateContext update_context;
  MemoryAllocator memory_allocator{MemoryAllocator(0, nullptr)};

  BackendInitContext init_context(&memory_allocator);
  
  // Create backend option
  options.set_option(StrKey("Backend"), "GPU");
  
  // Update before init
  Error err = mock_backend->update(update_context, options.view());
  EXPECT_EQ(err, Error::Ok);
    
  // Now call init
  FreeableBuffer* processed = nullptr;  // Not used in mock
  ArrayRef<CompileSpec> compile_specs;  // Empty
  auto handle_or_error = mock_backend->init(init_context, processed, compile_specs);
  EXPECT_EQ(handle_or_error.error(), Error::Ok);
  
  // Verify state
  EXPECT_TRUE(mock_backend->init_called);
  EXPECT_EQ(mock_backend->update_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 0);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "GPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateAfterInitBeforeExecute) {
  BackendUpdateContext update_context;
  MemoryAllocator init_memory_allocator{MemoryAllocator(0, nullptr)};
  BackendInitContext init_context(&init_memory_allocator);
  BackendExecutionContext execute_context;
  
  // First call init
  FreeableBuffer* processed = nullptr;
  ArrayRef<CompileSpec> compile_specs;
  auto handle_or_error = mock_backend->init(init_context, processed, compile_specs);
  EXPECT_TRUE(handle_or_error.ok());
  
  // Verify init called but execute not called
  EXPECT_TRUE(mock_backend->init_called);
  EXPECT_EQ(mock_backend->execute_count, 0);
  
  // Now update
  options.set_option(StrKey("Backend"), "CPU");
  Error err = mock_backend->update(update_context, options.view());
  EXPECT_EQ(err, Error::Ok);
  
  // Now execute
  DelegateHandle* handle = handle_or_error.get();
  EValue** args = nullptr;  // Not used in mock
  err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);
  
  // Verify state
  EXPECT_EQ(mock_backend->update_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 1);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "CPU");
}

TEST_F(BackendInterfaceUpdateTest, UpdateBetweenExecutes) {
  BackendUpdateContext update_context;
  MemoryAllocator init_memory_allocator{MemoryAllocator(0, nullptr)};
  BackendInitContext init_context(&init_memory_allocator);
  BackendExecutionContext execute_context;
  
  // Initialize
  FreeableBuffer* processed = nullptr;
  ArrayRef<CompileSpec> compile_specs;
  auto handle_or_error = mock_backend->init(init_context, processed, compile_specs);
  EXPECT_TRUE(handle_or_error.ok());
  DelegateHandle* handle = handle_or_error.get();
  
  // First execute
  EValue** args = nullptr;
  Error err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);
  
  // Update between executes
  options.set_option(StrKey("Backend"), "NPU");
  err = mock_backend->update(update_context, options.view());
  EXPECT_EQ(err, Error::Ok);
  
  // Second execute
  err = mock_backend->execute(execute_context, handle, args);
  EXPECT_EQ(err, Error::Ok);
  
  // Verify state
  EXPECT_EQ(mock_backend->update_count, 1);
  EXPECT_EQ(mock_backend->execute_count, 2);
  ASSERT_TRUE(mock_backend->target_backend.has_value());
  EXPECT_STREQ(mock_backend->target_backend.value().c_str(), "NPU");
}
