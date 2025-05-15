/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstddef>
#include <memory>

// Template struct to hold tensor data and tensor
template <typename T>
struct TensorStruct {
  std::unique_ptr<executorch::aten::TensorImpl> tensor;
  T* data;
  // data size in bytes
  size_t size;
};
