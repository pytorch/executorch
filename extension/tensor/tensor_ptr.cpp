/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch {
namespace extension {

runtime::Error resize_tensor_ptr(
    TensorPtr& tensor,
    const std::vector<exec_aten::SizesType>& sizes) {
  return runtime::resize_tensor(
      *tensor,
      exec_aten::ArrayRef<exec_aten::SizesType>(sizes.data(), sizes.size()));
}

} // namespace extension
} // namespace executorch
