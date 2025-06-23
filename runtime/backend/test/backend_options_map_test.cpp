/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/backend/options_map.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::BackendOptionsMap;
using executorch::runtime::Error;
using executorch::runtime::OptionKey;

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
} // namespace runtime
} // namespace executorch
