/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_options.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/backend_update.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOptions;
using executorch::runtime::BackendOptionsMap;
using executorch::runtime::BackendUpdateContext;
using executorch::runtime::BoolKey;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::IntKey;
using executorch::runtime::OptionKey;
using executorch::runtime::register_backend;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::StrKey;

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
      EValue** args) const override {
    return Error::Ok;
  }

  Error get_option(
      BackendUpdateContext& context,
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
      BackendUpdateContext& context,
      const Span<executorch::runtime::BackendOption>& backend_options) override {
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
TEST_F(BackendUpdateTest, TestSetOption) {
  BackendOptionsMap<3> map;
  BackendOptions<1> backend_options;
  int new_num_threads = 4;
  backend_options.set_option(IntKey("NumberOfThreads"), new_num_threads);
  map.add("StubBackend", backend_options.view());

  auto status = set_option(map.entries());
  ASSERT_EQ(status, Error::Ok);

  // Verify the map contains the expected data
  ASSERT_EQ(map.size(), 1);
  auto options = map.get("StubBackend");
  ASSERT_EQ(options.size(), 1);

  // Verify that the backend actually received the options
  ASSERT_EQ(stub_backend->last_options_size, 1);
  ASSERT_EQ(stub_backend->last_num_threads, new_num_threads);
}

// Test get_option functionality
TEST_F(BackendUpdateTest, TestGetOption) {
  // First, set some options in the backend
  BackendOptionsMap<3> set_map;
  BackendOptions<1> set_backend_options;
  int expected_num_threads = 8;
  set_backend_options.set_option(
      IntKey("NumberOfThreads"), expected_num_threads);
  set_map.add("StubBackend", set_backend_options.view());

  auto set_status = set_option(set_map.entries());
  ASSERT_EQ(set_status, Error::Ok);
  ASSERT_EQ(stub_backend->last_num_threads, expected_num_threads);

  // Reset get_option tracking variables
  stub_backend->get_option_called = false;
  stub_backend->get_option_call_count = 0;
  stub_backend->found_expected_key = false;

  // Now create a map with options for get_option to process
  BackendOptionsMap<3> get_map;
  BackendOptions<1> get_backend_options;
  get_backend_options.set_option(IntKey("NumberOfThreads"), 0);
  get_map.add("StubBackend", get_backend_options.view());

  // Call get_option to test the API
  auto get_status = get_option(get_map.entries());
  ASSERT_EQ(get_status, Error::Ok);

  ASSERT_TRUE(
      std::get<int>(get_map.entries()[0].options[0].value) ==
      expected_num_threads);

  // // Verify that the backend's get_option method was called correctly
  // ASSERT_TRUE(stub_backend->get_option_called);
  // ASSERT_EQ(stub_backend->get_option_call_count, 1);
  // ASSERT_EQ(stub_backend->last_get_option_size, 1);
  // ASSERT_TRUE(stub_backend->found_expected_key);
}
