/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace executorch::backends::aoti::slim {

inline SlimTensor scalar_to_tensor(
    const executorch::backends::aoti::slim::c10::Scalar& s,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor result = empty_strided({}, {}, s.type(), device);
  result.fill_(s);
  return result;
}

} // namespace executorch::backends::aoti::slim
