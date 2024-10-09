/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

Error sort_tensor(
    const Tensor& tensor,
    Tensor& sorted_tensor,
    Tensor& sorted_indice,
    bool descending = false);

} // namespace executor
} // namespace torch
