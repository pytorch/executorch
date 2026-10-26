/*
 * Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "NeuronAdapterLibrarySelector.h"
#include "NeuronError.h"

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace backends {
namespace neuron {

struct NeuronAdapterInitResult {
  runtime::Error error = runtime::Error::Ok;
  int requestedMajor = 0;
  NeuronRuntimeVersion runtimeVersion{};
  int versionStatus = NEURON_NO_ERROR;
  bool versionQueried = false;
};

template <typename ConfigureAdapter, typename GetVersion>
NeuronAdapterInitResult validateNeuronAdapterForInit(
    runtime::ArrayRef<runtime::CompileSpec> compileSpecs,
    ConfigureAdapter configureAdapter,
    GetVersion getVersion) {
  NeuronAdapterInitResult result;
  result.requestedMajor = scanAdapterLibraryMajor(compileSpecs);
  if (result.requestedMajor < 0) {
    result.error = runtime::Error::InvalidProgram;
    return result;
  }
  if (!configureAdapter(result.requestedMajor)) {
    result.error = runtime::Error::DelegateInvalidCompatibility;
    return result;
  }
  if (result.requestedMajor == 0) {
    return result;
  }

  result.versionQueried = true;
  result.versionStatus = getVersion(&result.runtimeVersion);
  if (result.versionStatus != NEURON_NO_ERROR ||
      result.runtimeVersion.major != result.requestedMajor) {
    result.error = runtime::Error::DelegateInvalidCompatibility;
  }
  return result;
}

struct NeuronNetworkLoadResult {
  int status;
  runtime::Error error;
};

template <typename LoadCompiledNetwork, typename Cleanup>
NeuronNetworkLoadResult loadCompiledNetworkForInit(
    LoadCompiledNetwork loadCompiledNetwork,
    Cleanup cleanup) {
  const int status = loadCompiledNetwork();
  const runtime::Error error = mapNeuronLoadError(status);
  if (error != runtime::Error::Ok) {
    cleanup();
  }
  return {status, error};
}

} // namespace neuron
} // namespace backends
} // namespace executorch
