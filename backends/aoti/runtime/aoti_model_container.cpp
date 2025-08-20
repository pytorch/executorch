/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "aoti_model_container.h"

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

// Global function pointers for AOT Inductor model container operations
// These will be loaded dynamically from the shared library
AOTInductorModelContainerCreateWithDeviceFunc
    AOTInductorModelContainerCreateWithDevice = nullptr;
AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete = nullptr;
AOTInductorModelContainerGetNumInputsFunc
    AOTInductorModelContainerGetNumInputs = nullptr;
AOTInductorModelContainerGetInputNameFunc
    AOTInductorModelContainerGetInputName = nullptr;
AOTInductorModelContainerGetNumConstantsFunc
    AOTInductorModelContainerGetNumConstants = nullptr;
AOTInductorModelContainerGetNumOutputsFunc
    AOTInductorModelContainerGetNumOutputs = nullptr;
AOTInductorModelContainerRunFunc AOTInductorModelContainerRun = nullptr;

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
