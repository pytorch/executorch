/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

/**
 * Creates and owns the necessary metadata for a Tensor instance. Does not own
 * the data pointer.
 */
class ManagedTensor {
 public:
  ManagedTensor(
      void* data,
      const std::vector<exec_aten::SizesType>& sizes,
      exec_aten::ScalarType dtype)
      : sizes_(sizes),
        tensor_impl_(
            /*type=*/dtype,
            /*dim=*/sizes_.size(),
            /*sizes=*/sizes_.data(),
            /*data=*/data,
            /*dim_order=*/nullptr,
            /*strides=*/nullptr,
            /*dynamism=*/
            executorch::runtime::TensorShapeDynamism::DYNAMIC_BOUND) {}

  /**
   * Get the Tensor object managed by this class.
   */
  exec_aten::Tensor get_tensor() {
    return exec_aten::Tensor(&tensor_impl_);
  }

 private:
  std::vector<exec_aten::SizesType> sizes_;
  exec_aten::TensorImpl tensor_impl_;
};
