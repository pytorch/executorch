/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/backend/options_map.h>
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
using executorch::runtime::BackendOptionsMap;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::OptionKey;
using executorch::runtime::register_backend;
using executorch::runtime::Result;

namespace executorch {
namespace runtime {

class BackendOptionsMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize any necessary runtime components
    executorch::runtime::runtime_init();
  }
  // Assume 3 backends, each with max 5 options
  BackendOptionsMap<3> map;
};

TEST_F(BackendOptionsMapTest, BasicAddAndRetrieve) {
  BackendOptions<5> cpu_options;

  cpu_options.set_option("use_fp16", true);
  cpu_options.set_option("thead", 4);
  map.add("CPU", cpu_options.view());

  auto retrieved = map.get("CPU");
  EXPECT_GE(retrieved.size(), 1);

  // bool value;
  bool found = false;
  for (auto retrieved_option : retrieved) {
    if (strcmp(retrieved_option.key, "use_fp16") == 0) {
      EXPECT_EQ(std::get<bool>(retrieved_option.value), true);
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(BackendOptionsMapTest, CapacityLimits) {
  BackendOptionsMap<2> small_map; // Only 2 backends capacity

  BackendOptions<5> options;
  ASSERT_EQ(small_map.add("CPU", options.view()), Error::Ok);
  ASSERT_EQ(small_map.add("GPU", options.view()), Error::Ok);
  // Return error if it exceeds capacity
  ASSERT_EQ(small_map.add("NPU", options.view()), Error::InvalidArgument);
}

TEST_F(BackendOptionsMapTest, EntryIteration) {
  BackendOptions<2> cpu_options;
  BackendOptions<3> gpu_options;

  // Add to map using Span
  ASSERT_EQ(map.add("CPU", cpu_options.view()), Error::Ok);
  ASSERT_EQ(map.add("GPU", gpu_options.view()), Error::Ok);

  auto entries = map.entries();
  // Should have 2 backends (entries)
  ASSERT_EQ(entries.size(), 2);

  bool found_cpu = false;
  bool found_gpu = false;
  for (const auto& entry : entries) {
    if (strcmp(entry.backend_name, "CPU") == 0)
      found_cpu = true;
    if (strcmp(entry.backend_name, "GPU") == 0)
      found_gpu = true;
  }
  // Should find CPU and GPU in the entries
  EXPECT_TRUE(found_cpu);
  EXPECT_TRUE(found_gpu);
}

TEST_F(BackendOptionsMapTest, ConstCorrectness) {
  auto cpu_options = BackendOptions<5>();
  ASSERT_EQ(map.add("CPU", cpu_options.view()), Error::Ok);

  const auto& const_map = map;
  auto options_retrived = const_map.get("CPU");
  EXPECT_EQ(options_retrived.size(), 0);

  auto entries = const_map.entries();
  EXPECT_FALSE(entries.empty());
}

TEST_F(BackendOptionsMapTest, EmptyMapBehavior) {
  EXPECT_EQ(map.get("CPU").size(), 0);
  EXPECT_TRUE(map.entries().empty());
  EXPECT_EQ(map.entries().size(), 0);
}

TEST_F(BackendOptionsMapTest, OptionIsolation) {
  BackendOptions<2> cpu_options;
  cpu_options.set_option("Debug", true);
  cpu_options.set_option("NumThreads", 3);

  BackendOptions<3> gpu_options;
  gpu_options.set_option("Profile", true);
  gpu_options.set_option("Mem", 1024);
  gpu_options.set_option("Hardware", "H100");

  // Add to map using Span
  map.add("CPU", cpu_options.view());
  map.add("GPU", gpu_options.view());

  // Test CPU options
  auto cpu_opts = map.get("CPU");
  ASSERT_FALSE(cpu_opts.empty());

  // Verify CPU has its own option
  EXPECT_EQ(cpu_opts.size(), 2);
  EXPECT_STREQ(cpu_opts[0].key, "Debug");
  EXPECT_EQ(std::get<bool>(cpu_opts[0].value), true);
  EXPECT_STREQ(cpu_opts[1].key, "NumThreads");
  EXPECT_EQ(std::get<int>(cpu_opts[1].value), 3);

  // Test GPU options
  auto gpu_opts = map.get("GPU");
  ASSERT_FALSE(gpu_opts.empty());

  // Verify GPU has its own option
  EXPECT_EQ(gpu_opts.size(), 3);
  EXPECT_STREQ(gpu_opts[0].key, "Profile");
  EXPECT_EQ(std::get<bool>(gpu_opts[0].value), true);
  EXPECT_STREQ(gpu_opts[1].key, "Mem");
  EXPECT_EQ(std::get<int>(gpu_opts[1].value), 1024);
  EXPECT_STREQ(gpu_opts[2].key, "Hardware");
  EXPECT_STREQ(std::get<const char*>(gpu_opts[2].value), "H100");
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
      EValue** args) const override {
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
      const Span<executorch::runtime::BackendOption>& backend_options)
      override {
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
  backend_options.set_option("NumberOfThreads", new_num_threads);
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
  set_backend_options.set_option("NumberOfThreads", expected_num_threads);
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
  get_backend_options.set_option("NumberOfThreads", 0);
  get_map.add("StubBackend", get_backend_options.view());

  // Call get_option to test the API
  auto get_status = get_option(get_map.entries());
  ASSERT_EQ(get_status, Error::Ok);

  ASSERT_TRUE(
      std::get<int>(get_map.entries()[0].options[0].value) ==
      expected_num_threads);

  // Verify that the backend's get_option method was called correctly
  ASSERT_TRUE(stub_backend->get_option_called);
  ASSERT_EQ(stub_backend->get_option_call_count, 1);
  ASSERT_EQ(stub_backend->last_get_option_size, 1);
  ASSERT_TRUE(stub_backend->found_expected_key);
}
} // namespace runtime
} // namespace executorch
