/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "api/NeuronAdapter.h"

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace backends {
namespace neuron {

constexpr runtime::Error mapNeuronLoadError(int status) {
  switch (status) {
    case NEURON_NO_ERROR:
      return runtime::Error::Ok;
    case -1:
    case NEURON_BAD_VERSION:
      return runtime::Error::DelegateInvalidCompatibility;
    case NEURON_OUT_OF_MEMORY:
      return runtime::Error::DelegateMemoryAllocationFailed;
    case NEURON_BAD_DATA:
      return runtime::Error::InvalidProgram;
    case NEURON_UNMAPPABLE:
      return runtime::Error::AccessFailed;
    case NEURON_BAD_STATE:
    case NEURON_DEAD_OBJECT:
      return runtime::Error::InvalidState;
    case NEURON_UNAVAILABLE_DEVICE:
      return runtime::Error::NotFound;
    case NEURON_MISSED_DEADLINE_TRANSIENT:
    case NEURON_MISSED_DEADLINE_PERSISTENT:
    case NEURON_RESOURCE_EXHAUSTED_TRANSIENT:
    case NEURON_RESOURCE_EXHAUSTED_PERSISTENT:
      return runtime::Error::OutOfResources;
    case NEURON_INCOMPLETE:
    case NEURON_UNEXPECTED_NULL:
    case NEURON_OP_FAILED:
    case NEURON_OUTPUT_INSUFFICIENT_SIZE:
    default:
      return runtime::Error::Internal;
  }
}

} // namespace neuron
} // namespace backends
} // namespace executorch
