/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

/*
 * Repeats in tensor along the dimensions specified in repeats.
 *
 * @param[in] Input tensor that we want to repeat.
 * @param[in] The number of times to repeat this tensor along each dimension
 * @param[in] Output tensor to write to.
 *
 * @returns The status of the repeat operation.
 */
Error repeat_tensor(
    const exec_aten::Tensor& in,
    exec_aten::ArrayRef<int64_t> repeats,
    exec_aten::Tensor& out);

} // namespace executor
} // namespace torch
