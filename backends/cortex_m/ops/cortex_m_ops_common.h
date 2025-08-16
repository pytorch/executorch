/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Scalar = torch::executor::Scalar;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using Error = executorch::runtime::Error;

inline void validate_quantized_inputs(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output) {
  ET_CHECK_MSG(input1.scalar_type() == ScalarType::Char, "Input1 must be int8");
  ET_CHECK_MSG(input2.scalar_type() == ScalarType::Char, "Input2 must be int8");
  ET_CHECK_MSG(output.scalar_type() == ScalarType::Char, "Output must be int8");
  ET_CHECK_MSG(
      input1.sizes() == input2.sizes(), "Input tensors must be the same shape");
  ET_CHECK_MSG(
      input1.scalar_type() == input2.scalar_type(),
      "Input tensors must be the same dtype");
  ET_CHECK_MSG(
      (torch::executor::resize_to_broadcast_target_size(
           input1, input2, output) == Error::Ok),
      "Broadcast error: resize_to_broadcast_target_size failed");
  return;
}
