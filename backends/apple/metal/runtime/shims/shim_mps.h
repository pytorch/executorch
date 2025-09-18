/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "types.h"

namespace executorch {
namespace backends {
namespace aoti {

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to match PyTorch's AOTI MPS interface
struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

// Match PyTorch's AtenTensorHandle definition
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

/**
 * ExecutorTorch implementation of aoti_torch_mps_set_arg_tensor.
 * This replaces PyTorch's MPS implementation to work with ExecutorTorch tensors
 * that wrap Metal memory but don't have PyTorch device metadata.
 */
AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

/**
 * ExecutorTorch implementation of aoti_torch_mps_set_arg_int.
 * This replaces PyTorch's MPS implementation for setting integer arguments.
 */
AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace aoti
} // namespace backends
} // namespace executorch
