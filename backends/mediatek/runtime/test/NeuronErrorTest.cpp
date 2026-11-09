/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "api/NeuronError.h"

#include <array>

#include <gtest/gtest.h>

namespace executorch {
namespace backends {
namespace neuron {
namespace {

struct ErrorMapping {
  int status;
  runtime::Error error;
};

TEST(NeuronErrorTest, MapsEveryDocumentedLoadStatus) {
  constexpr std::array<ErrorMapping, 16> mappings{{
      {NEURON_NO_ERROR, runtime::Error::Ok},
      {NEURON_OUT_OF_MEMORY, runtime::Error::DelegateMemoryAllocationFailed},
      {NEURON_INCOMPLETE, runtime::Error::Internal},
      {NEURON_UNEXPECTED_NULL, runtime::Error::Internal},
      {NEURON_BAD_DATA, runtime::Error::InvalidProgram},
      {NEURON_OP_FAILED, runtime::Error::Internal},
      {NEURON_UNMAPPABLE, runtime::Error::AccessFailed},
      {NEURON_BAD_STATE, runtime::Error::InvalidState},
      {NEURON_BAD_VERSION, runtime::Error::DelegateInvalidCompatibility},
      {NEURON_OUTPUT_INSUFFICIENT_SIZE, runtime::Error::Internal},
      {NEURON_UNAVAILABLE_DEVICE, runtime::Error::NotFound},
      {NEURON_MISSED_DEADLINE_TRANSIENT, runtime::Error::OutOfResources},
      {NEURON_MISSED_DEADLINE_PERSISTENT, runtime::Error::OutOfResources},
      {NEURON_RESOURCE_EXHAUSTED_TRANSIENT, runtime::Error::OutOfResources},
      {NEURON_RESOURCE_EXHAUSTED_PERSISTENT, runtime::Error::OutOfResources},
      {NEURON_DEAD_OBJECT, runtime::Error::InvalidState},
  }};

  for (const auto& mapping : mappings) {
    EXPECT_EQ(mapNeuronLoadError(mapping.status), mapping.error)
        << "Neuron status " << mapping.status;
  }
}

TEST(NeuronErrorTest, MapsAdapterLoadAndUnknownFailures) {
  EXPECT_EQ(
      mapNeuronLoadError(-1), runtime::Error::DelegateInvalidCompatibility);
  EXPECT_EQ(mapNeuronLoadError(99), runtime::Error::Internal);
}

} // namespace
} // namespace neuron
} // namespace backends
} // namespace executorch
