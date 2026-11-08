/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "api/NeuronBackendInit.h"

#include <array>

#include <gtest/gtest.h>

namespace executorch {
namespace backends {
namespace neuron {
namespace {

TEST(NeuronBackendInitTest, InitBadLabelFailsBeforeAdapterConfiguration) {
  std::array<char, 2> value{'9', 'x'};
  std::array<runtime::CompileSpec, 1> compileSpecs{{
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
  }};
  bool configureCalled = false;
  bool getVersionCalled = false;

  const auto result = validateNeuronAdapterForInit(
      compileSpecs,
      [&](int) {
        configureCalled = true;
        return true;
      },
      [&](NeuronRuntimeVersion*) {
        getVersionCalled = true;
        return NEURON_NO_ERROR;
      });

  EXPECT_EQ(result.error, runtime::Error::InvalidProgram);
  EXPECT_FALSE(configureCalled);
  EXPECT_FALSE(getVersionCalled);
}

TEST(NeuronBackendInitTest, InitVersionMismatchFailsCompatibilityCheck) {
  std::array<char, 1> value{'9'};
  std::array<runtime::CompileSpec, 1> compileSpecs{{
      {kNeuronAdapterMajorKey.data(), {value.data(), value.size()}},
  }};

  const auto result = validateNeuronAdapterForInit(
      compileSpecs,
      [](int requestedMajor) { return requestedMajor == 9; },
      [](NeuronRuntimeVersion* version) {
        version->major = 8;
        return NEURON_NO_ERROR;
      });

  EXPECT_EQ(result.error, runtime::Error::DelegateInvalidCompatibility);
  EXPECT_TRUE(result.versionQueried);
  EXPECT_EQ(result.requestedMajor, 9);
  EXPECT_EQ(result.runtimeVersion.major, 8);
}

TEST(NeuronBackendInitTest, InitNetworkLoadFailureCleansUpAndReturnsError) {
  int loadCalls = 0;
  int cleanupCalls = 0;

  const auto result = loadCompiledNetworkForInit(
      [&]() {
        ++loadCalls;
        return NEURON_OUT_OF_MEMORY;
      },
      [&]() { ++cleanupCalls; });

  EXPECT_EQ(loadCalls, 1);
  EXPECT_EQ(cleanupCalls, 1);
  EXPECT_EQ(result.status, NEURON_OUT_OF_MEMORY);
  EXPECT_EQ(result.error, runtime::Error::DelegateMemoryAllocationFailed);
}

TEST(NeuronBackendInitTest, InitNetworkLoadSuccessKeepsDelegateAlive) {
  int cleanupCalls = 0;

  const auto result = loadCompiledNetworkForInit(
      []() { return NEURON_NO_ERROR; }, [&]() { ++cleanupCalls; });

  EXPECT_EQ(cleanupCalls, 0);
  EXPECT_EQ(result.status, NEURON_NO_ERROR);
  EXPECT_EQ(result.error, runtime::Error::Ok);
}

} // namespace
} // namespace neuron
} // namespace backends
} // namespace executorch
