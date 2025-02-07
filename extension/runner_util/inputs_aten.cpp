/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <vector>

#include <ATen/ATen.h> // @manual=//caffe2/aten:ATen-core
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>

using executorch::runtime::Error;
using executorch::runtime::Method;
using executorch::runtime::TensorInfo;

namespace executorch {
namespace extension {
namespace internal {

Error fill_and_set_input(
    Method& method,
    TensorInfo& tensor_meta,
    size_t input_index,
    void* data_ptr) {
  // Convert the sizes array from int32_t to int64_t.
  std::vector<int64_t> sizes;
  for (auto s : tensor_meta.sizes()) {
    sizes.push_back(s);
  }
  at::Tensor t = at::from_blob(
      data_ptr, sizes, at::TensorOptions(tensor_meta.scalar_type()));
  t.fill_(1.0f);

  return method.set_input(t, input_index);
}

} // namespace internal
} // namespace extension
} // namespace executorch
