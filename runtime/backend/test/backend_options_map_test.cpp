/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::kMaxOptionValueLength;
using executorch::runtime::LoadBackendOptionsMap;
using executorch::runtime::Span;

class LoadBackendOptionsMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// Test default constructor creates empty map
TEST_F(LoadBackendOptionsMapTest, DefaultConstructorCreatesEmptyMap) {
  LoadBackendOptionsMap map;
  EXPECT_EQ(map.size(), 0);
  EXPECT_FALSE(map.has_options("CoreMLBackend"));
}

// Test set_options and get_options round trip
TEST_F(LoadBackendOptionsMapTest, SetAndGetOptionsRoundTrip) {
  BackendOptions<4> coreml_opts;
  coreml_opts.set_option("compute_unit", "cpu_and_gpu");
  coreml_opts.set_option("num_threads", 4);

  LoadBackendOptionsMap map;
  EXPECT_EQ(map.set_options("CoreMLBackend", coreml_opts.view()), Error::Ok);

  auto retrieved = map.get_options("CoreMLBackend");
  EXPECT_EQ(retrieved.size(), 2);

  // Verify we can read the options back
  const char* compute_unit = nullptr;
  int num_threads = 0;
  for (size_t i = 0; i < retrieved.size(); ++i) {
    const auto& opt = retrieved[i];
    if (strcmp(opt.key, "compute_unit") == 0) {
      if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(
              &opt.value)) {
        compute_unit = arr->data();
      }
    } else if (strcmp(opt.key, "num_threads") == 0) {
      if (auto* val = std::get_if<int>(&opt.value)) {
        num_threads = *val;
      }
    }
  }
  EXPECT_STREQ(compute_unit, "cpu_and_gpu");
  EXPECT_EQ(num_threads, 4);
}

// Test has_options returns correct values
TEST_F(LoadBackendOptionsMapTest, HasOptionsReturnsCorrectValues) {
  BackendOptions<2> opts;
  opts.set_option("key", "value");

  LoadBackendOptionsMap map;
  EXPECT_FALSE(map.has_options("CoreMLBackend"));

  map.set_options("CoreMLBackend", opts.view());
  EXPECT_TRUE(map.has_options("CoreMLBackend"));
  EXPECT_FALSE(map.has_options("XNNPACKBackend"));
}

// Test get_options returns empty span for unknown backend
TEST_F(LoadBackendOptionsMapTest, GetOptionsReturnsEmptyForUnknownBackend) {
  LoadBackendOptionsMap map;
  auto opts = map.get_options("UnknownBackend");
  EXPECT_EQ(opts.size(), 0);
}

// Test multiple backends
TEST_F(LoadBackendOptionsMapTest, MultipleBackends) {
  BackendOptions<2> coreml_opts;
  coreml_opts.set_option("compute_unit", "cpu_and_ne");

  BackendOptions<2> xnnpack_opts;
  xnnpack_opts.set_option("num_threads", 8);

  LoadBackendOptionsMap map;
  EXPECT_EQ(map.set_options("CoreMLBackend", coreml_opts.view()), Error::Ok);
  EXPECT_EQ(map.set_options("XNNPACKBackend", xnnpack_opts.view()), Error::Ok);

  EXPECT_EQ(map.size(), 2);
  EXPECT_TRUE(map.has_options("CoreMLBackend"));
  EXPECT_TRUE(map.has_options("XNNPACKBackend"));

  auto coreml_retrieved = map.get_options("CoreMLBackend");
  auto xnnpack_retrieved = map.get_options("XNNPACKBackend");

  EXPECT_EQ(coreml_retrieved.size(), 1);
  EXPECT_EQ(xnnpack_retrieved.size(), 1);
}

// Test updating existing backend options
TEST_F(LoadBackendOptionsMapTest, UpdateExistingBackendOptions) {
  BackendOptions<2> opts_v1;
  opts_v1.set_option("compute_unit", "cpu_only");

  BackendOptions<2> opts_v2;
  opts_v2.set_option("compute_unit", "cpu_and_gpu");
  opts_v2.set_option("enable_profiling", true);

  LoadBackendOptionsMap map;
  map.set_options("CoreMLBackend", opts_v1.view());
  EXPECT_EQ(map.get_options("CoreMLBackend").size(), 1);

  // Update with new options
  map.set_options("CoreMLBackend", opts_v2.view());
  EXPECT_EQ(map.size(), 1); // Still only one backend
  EXPECT_EQ(map.get_options("CoreMLBackend").size(), 2); // But now 2 options
}

// Test max backends limit
TEST_F(LoadBackendOptionsMapTest, MaxBackendsLimit) {
  LoadBackendOptionsMap map;
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  // Add 8 backends (the limit)
  const char* backend_ids[] = {
      "Backend1",
      "Backend2",
      "Backend3",
      "Backend4",
      "Backend5",
      "Backend6",
      "Backend7",
      "Backend8"};

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(map.set_options(backend_ids[i], opts.view()), Error::Ok);
  }

  EXPECT_EQ(map.size(), 8);

  // Adding a 9th backend should fail
  EXPECT_EQ(map.set_options("Backend9", opts.view()), Error::InvalidArgument);
  EXPECT_EQ(map.size(), 8);
}

// Test null backend_id handling
TEST_F(LoadBackendOptionsMapTest, NullBackendIdHandling) {
  LoadBackendOptionsMap map;
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  // set_options with null should fail
  EXPECT_EQ(map.set_options(nullptr, opts.view()), Error::InvalidArgument);

  // get_options with null should return empty span
  auto result = map.get_options(nullptr);
  EXPECT_EQ(result.size(), 0);

  // has_options with null should return false
  EXPECT_FALSE(map.has_options(nullptr));
}

// Test empty backend_id handling
TEST_F(LoadBackendOptionsMapTest, EmptyBackendIdHandling) {
  LoadBackendOptionsMap map;
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  // set_options with empty string should fail
  EXPECT_EQ(map.set_options("", opts.view()), Error::InvalidArgument);
}

// Test empty options span
TEST_F(LoadBackendOptionsMapTest, EmptyOptionsSpan) {
  LoadBackendOptionsMap map;
  BackendOptions<1> empty_opts;

  // Should be able to set empty options
  EXPECT_EQ(map.set_options("CoreMLBackend", empty_opts.view()), Error::Ok);
  EXPECT_TRUE(map.has_options("CoreMLBackend"));

  auto retrieved = map.get_options("CoreMLBackend");
  EXPECT_EQ(retrieved.size(), 0);
}

// Test long backend_id is rejected
TEST_F(LoadBackendOptionsMapTest, LongBackendIdRejected) {
  LoadBackendOptionsMap map;
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  // Create a backend ID that is exactly at the limit (64 chars including null)
  // This should fail because id_len >= kMaxBackendIdLength
  char long_id[65];
  memset(long_id, 'A', 64);
  long_id[64] = '\0';

  EXPECT_EQ(map.set_options(long_id, opts.view()), Error::InvalidArgument);
  EXPECT_EQ(map.size(), 0);

  // A backend ID of 63 chars (plus null = 64 total) should succeed
  char max_valid_id[64];
  memset(max_valid_id, 'B', 63);
  max_valid_id[63] = '\0';

  EXPECT_EQ(map.set_options(max_valid_id, opts.view()), Error::Ok);
  EXPECT_EQ(map.size(), 1);
  EXPECT_TRUE(map.has_options(max_valid_id));
}

/**
 * Example backend options builder demonstrating the recommended pattern.
 *
 * Backend developers should follow this pattern to create type-safe option
 * builders for their backends. Key requirements:
 *
 * 1. Provide a static backend_id() method returning the backend identifier
 * 2. Provide a view() method returning Span<BackendOption>
 * 3. Use an enum for options with a fixed set of valid values
 * 4. Provide type-safe setter methods that return *this for chaining
 *
 * This enables usage like:
 * @code
 *   ExampleBackendOptions opts;
 *   opts.setNumThreads(4).setEnableOptimization(true);
 *   map.set_options(opts);  // Uses the template overload
 * @endcode
 *
 * See coreml_backend_options.h for a real-world example.
 */
class ExampleBackendOptions {
 public:
  // Option 1: Enum-based option with type safety
  enum class Precision { FLOAT32, FLOAT16, INT8 };

  ExampleBackendOptions& setPrecision(Precision p) {
    const char* value = nullptr;
    switch (p) {
      case Precision::FLOAT32:
        value = "float32";
        break;
      case Precision::FLOAT16:
        value = "float16";
        break;
      case Precision::INT8:
        value = "int8";
        break;
    }
    options_.set_option("precision", value);
    return *this;
  }

  // Option 2: Integer option
  ExampleBackendOptions& setNumThreads(int num_threads) {
    options_.set_option("num_threads", num_threads);
    return *this;
  }

  // Option 3: Boolean option
  ExampleBackendOptions& setEnableOptimization(bool enable) {
    options_.set_option("enable_optimization", enable);
    return *this;
  }

  // Required: Returns the backend identifier
  static constexpr const char* backend_id() {
    return "ExampleBackend";
  }

  // Required: Returns a view of the configured options
  Span<BackendOption> view() {
    return options_.view();
  }

 private:
  BackendOptions<8> options_;
};

// Test template set_options with builder
TEST_F(LoadBackendOptionsMapTest, SetOptionsWithBuilder) {
  LoadBackendOptionsMap map;

  // Example of fluent builder API usage
  ExampleBackendOptions builder;
  builder.setPrecision(ExampleBackendOptions::Precision::FLOAT16)
      .setNumThreads(4)
      .setEnableOptimization(true);

  EXPECT_EQ(map.set_options(builder), Error::Ok);
  EXPECT_EQ(map.size(), 1);
  EXPECT_TRUE(map.has_options("ExampleBackend"));

  auto retrieved = map.get_options("ExampleBackend");
  EXPECT_EQ(retrieved.size(), 3);

  // Verify we can read the options back
  const char* precision_value = nullptr;
  int num_threads_value = 0;
  bool enable_optimization_value = false;
  for (size_t i = 0; i < retrieved.size(); ++i) {
    const auto& opt = retrieved[i];
    if (strcmp(opt.key, "precision") == 0) {
      if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(
              &opt.value)) {
        precision_value = arr->data();
      }
    } else if (strcmp(opt.key, "num_threads") == 0) {
      if (auto* val = std::get_if<int>(&opt.value)) {
        num_threads_value = *val;
      }
    } else if (strcmp(opt.key, "enable_optimization") == 0) {
      if (auto* val = std::get_if<bool>(&opt.value)) {
        enable_optimization_value = *val;
      }
    }
  }
  EXPECT_STREQ(precision_value, "float16");
  EXPECT_EQ(num_threads_value, 4);
  EXPECT_TRUE(enable_optimization_value);
}

// Test template set_options updates existing backend
TEST_F(LoadBackendOptionsMapTest, SetOptionsWithBuilderUpdatesExisting) {
  LoadBackendOptionsMap map;

  // First set options via builder
  ExampleBackendOptions builder1;
  builder1.setNumThreads(4);
  EXPECT_EQ(map.set_options(builder1), Error::Ok);
  EXPECT_EQ(map.size(), 1);

  // Verify initial value
  auto retrieved1 = map.get_options("ExampleBackend");
  EXPECT_EQ(retrieved1.size(), 1);
  int num_threads1 = 0;
  if (auto* val = std::get_if<int>(&retrieved1[0].value)) {
    num_threads1 = *val;
  }
  EXPECT_EQ(num_threads1, 4);

  // Update via builder API with different value
  ExampleBackendOptions builder2;
  builder2.setNumThreads(8);
  EXPECT_EQ(map.set_options(builder2), Error::Ok);
  EXPECT_EQ(map.size(), 1); // Still only one backend

  // Verify value was updated
  auto retrieved2 = map.get_options("ExampleBackend");
  EXPECT_EQ(retrieved2.size(), 1);
  int num_threads2 = 0;
  if (auto* val = std::get_if<int>(&retrieved2[0].value)) {
    num_threads2 = *val;
  }
  EXPECT_EQ(num_threads2, 8); // Should be updated value
}
