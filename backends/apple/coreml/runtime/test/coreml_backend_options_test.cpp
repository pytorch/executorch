/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/coreml/runtime/include/coreml_backend/coreml_backend_options.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using executorch::backends::coreml::LoadOptionsBuilder;
using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::kMaxOptionValueLength;
using executorch::runtime::LoadBackendOptionsMap;

class CoreMLBackendOptionsTest : public ::testing::Test {
protected:
    void SetUp() override { executorch::runtime::runtime_init(); }
};

// Test default construction
TEST_F(CoreMLBackendOptionsTest, DefaultConstruction) {
    LoadOptionsBuilder builder;
    auto options = builder.view();
    EXPECT_EQ(options.size(), 0);
}

// Test backend_id returns correct value
TEST_F(CoreMLBackendOptionsTest, BackendId) { EXPECT_STREQ(LoadOptionsBuilder::backend_id(), "CoreMLBackend"); }

// Test setComputeUnit with CPU_ONLY
TEST_F(CoreMLBackendOptionsTest, SetComputeUnitCpuOnly) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_ONLY);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);
    EXPECT_STREQ(options[0].key, "compute_unit");

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "cpu_only");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test setComputeUnit with CPU_AND_GPU
TEST_F(CoreMLBackendOptionsTest, SetComputeUnitCpuAndGpu) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_GPU);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "cpu_and_gpu");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test setComputeUnit with CPU_AND_NE
TEST_F(CoreMLBackendOptionsTest, SetComputeUnitCpuAndNe) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_NE);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "cpu_and_ne");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test setComputeUnit with ALL
TEST_F(CoreMLBackendOptionsTest, SetComputeUnitAll) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::ALL);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "all");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test method chaining returns reference to builder
TEST_F(CoreMLBackendOptionsTest, MethodChaining) {
    LoadOptionsBuilder builder;
    auto& result = builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_GPU);

    // Should return reference to the same builder
    EXPECT_EQ(&result, &builder);
}

// Test integration with LoadBackendOptionsMap using template set_options
TEST_F(CoreMLBackendOptionsTest, IntegrationWithOptionsMap) {
    LoadOptionsBuilder coreml_opts;
    coreml_opts.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_NE);

    LoadBackendOptionsMap map;
    EXPECT_EQ(map.set_options(coreml_opts), Error::Ok);

    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(map.has_options("CoreMLBackend"));

    auto retrieved = map.get_options("CoreMLBackend");
    EXPECT_EQ(retrieved.size(), 1);
    EXPECT_STREQ(retrieved[0].key, "compute_unit");

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&retrieved[0].value)) {
        EXPECT_STREQ(arr->data(), "cpu_and_ne");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test that setting compute unit multiple times updates the value
TEST_F(CoreMLBackendOptionsTest, SetComputeUnitMultipleTimes) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_ONLY);
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::ALL);

    auto options = builder.view();
    // BackendOptions updates existing keys, so we should have 1 entry with the latest value
    EXPECT_EQ(options.size(), 1);

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "all"); // Last value wins
    } else {
        FAIL() << "Expected string value for compute_unit";
    }
}

// Test setCacheDirectory
TEST_F(CoreMLBackendOptionsTest, SetCacheDirectory) {
    LoadOptionsBuilder builder;
    builder.setCacheDirectory("/path/to/cache");

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);
    EXPECT_STREQ(options[0].key, "cache_dir");

    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "/path/to/cache");
    } else {
        FAIL() << "Expected string value for cache_dir";
    }
}

// Test setCacheDirectory method chaining
TEST_F(CoreMLBackendOptionsTest, SetCacheDirectoryChaining) {
    LoadOptionsBuilder builder;
    auto& result = builder.setCacheDirectory("/tmp/cache");

    // Should return reference to the same builder
    EXPECT_EQ(&result, &builder);
}

// Test combining setComputeUnit and setCacheDirectory
TEST_F(CoreMLBackendOptionsTest, CombinedOptions) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_NE).setCacheDirectory("/data/experiment_cache");

    auto options = builder.view();
    EXPECT_EQ(options.size(), 2);

    // Verify compute_unit
    EXPECT_STREQ(options[0].key, "compute_unit");
    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[0].value)) {
        EXPECT_STREQ(arr->data(), "cpu_and_ne");
    } else {
        FAIL() << "Expected string value for compute_unit";
    }

    // Verify cache_dir
    EXPECT_STREQ(options[1].key, "cache_dir");
    if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[1].value)) {
        EXPECT_STREQ(arr->data(), "/data/experiment_cache");
    } else {
        FAIL() << "Expected string value for cache_dir";
    }
}

// Test integration with LoadBackendOptionsMap including cache_dir
TEST_F(CoreMLBackendOptionsTest, IntegrationWithOptionsMapCacheDir) {
    LoadOptionsBuilder coreml_opts;
    coreml_opts.setComputeUnit(LoadOptionsBuilder::ComputeUnit::ALL).setCacheDirectory("/custom/cache/path");

    LoadBackendOptionsMap map;
    EXPECT_EQ(map.set_options(coreml_opts), Error::Ok);

    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(map.has_options("CoreMLBackend"));

    auto retrieved = map.get_options("CoreMLBackend");
    EXPECT_EQ(retrieved.size(), 2);

    // Find cache_dir option
    bool found_cache_dir = false;
    for (size_t i = 0; i < retrieved.size(); ++i) {
        if (std::strcmp(retrieved[i].key, "cache_dir") == 0) {
            found_cache_dir = true;
            if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&retrieved[i].value)) {
                EXPECT_STREQ(arr->data(), "/custom/cache/path");
            } else {
                FAIL() << "Expected string value for cache_dir";
            }
            break;
        }
    }
    EXPECT_TRUE(found_cache_dir) << "cache_dir option not found";
}

// Test setUseNewCache with true
TEST_F(CoreMLBackendOptionsTest, SetUseNewCacheTrue) {
    LoadOptionsBuilder builder;
    builder.setUseNewCache(true);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);
    EXPECT_STREQ(options[0].key, "_use_new_cache");

    if (auto* val = std::get_if<bool>(&options[0].value)) {
        EXPECT_TRUE(*val);
    } else {
        FAIL() << "Expected bool value for _use_new_cache";
    }
}

// Test setUseNewCache with false
TEST_F(CoreMLBackendOptionsTest, SetUseNewCacheFalse) {
    LoadOptionsBuilder builder;
    builder.setUseNewCache(false);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);
    EXPECT_STREQ(options[0].key, "_use_new_cache");

    if (auto* val = std::get_if<bool>(&options[0].value)) {
        EXPECT_FALSE(*val);
    } else {
        FAIL() << "Expected bool value for _use_new_cache";
    }
}

// Test setUseNewCache method chaining
TEST_F(CoreMLBackendOptionsTest, SetUseNewCacheChaining) {
    LoadOptionsBuilder builder;
    auto& result = builder.setUseNewCache(true);

    // Should return reference to the same builder
    EXPECT_EQ(&result, &builder);
}

// Test combining setComputeUnit, setCacheDirectory, and setUseNewCache
TEST_F(CoreMLBackendOptionsTest, AllOptionsCombined) {
    LoadOptionsBuilder builder;
    builder.setComputeUnit(LoadOptionsBuilder::ComputeUnit::CPU_AND_GPU)
        .setCacheDirectory("/path/to/cache")
        .setUseNewCache(true);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 3);

    // Find and verify each option
    bool found_compute_unit = false;
    bool found_cache_dir = false;
    bool found_use_new_cache = false;

    for (size_t i = 0; i < options.size(); ++i) {
        if (std::strcmp(options[i].key, "compute_unit") == 0) {
            found_compute_unit = true;
            if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[i].value)) {
                EXPECT_STREQ(arr->data(), "cpu_and_gpu");
            }
        } else if (std::strcmp(options[i].key, "cache_dir") == 0) {
            found_cache_dir = true;
            if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(&options[i].value)) {
                EXPECT_STREQ(arr->data(), "/path/to/cache");
            }
        } else if (std::strcmp(options[i].key, "_use_new_cache") == 0) {
            found_use_new_cache = true;
            if (auto* val = std::get_if<bool>(&options[i].value)) {
                EXPECT_TRUE(*val);
            }
        }
    }

    EXPECT_TRUE(found_compute_unit) << "compute_unit option not found";
    EXPECT_TRUE(found_cache_dir) << "cache_dir option not found";
    EXPECT_TRUE(found_use_new_cache) << "_use_new_cache option not found";
}

// Test integration with LoadBackendOptionsMap including _use_new_cache
TEST_F(CoreMLBackendOptionsTest, IntegrationWithOptionsMapUseNewCache) {
    LoadOptionsBuilder coreml_opts;
    coreml_opts.setUseNewCache(true);

    LoadBackendOptionsMap map;
    EXPECT_EQ(map.set_options(coreml_opts), Error::Ok);

    EXPECT_EQ(map.size(), 1);
    EXPECT_TRUE(map.has_options("CoreMLBackend"));

    auto retrieved = map.get_options("CoreMLBackend");
    EXPECT_EQ(retrieved.size(), 1);
    EXPECT_STREQ(retrieved[0].key, "_use_new_cache");

    if (auto* val = std::get_if<bool>(&retrieved[0].value)) {
        EXPECT_TRUE(*val);
    } else {
        FAIL() << "Expected bool value for _use_new_cache";
    }
}

// Test setUseNewCache updates when called multiple times
TEST_F(CoreMLBackendOptionsTest, SetUseNewCacheMultipleTimes) {
    LoadOptionsBuilder builder;
    builder.setUseNewCache(true);
    builder.setUseNewCache(false);

    auto options = builder.view();
    EXPECT_EQ(options.size(), 1);

    if (auto* val = std::get_if<bool>(&options[0].value)) {
        EXPECT_FALSE(*val); // Last value wins
    } else {
        FAIL() << "Expected bool value for _use_new_cache";
    }
}
