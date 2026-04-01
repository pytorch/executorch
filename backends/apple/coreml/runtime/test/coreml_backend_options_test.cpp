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
