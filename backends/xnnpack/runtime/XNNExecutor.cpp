/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

Error XNNExecutor::set_external_input(uint32_t id, Tensor* input) {
  externals_.emplace_back(xnn_external_value{id, input->data_ptr()});
  return Error::Ok;
}

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
