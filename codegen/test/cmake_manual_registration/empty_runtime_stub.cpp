/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/operator_registry.h>
#include <cstdlib>

namespace executorch {
namespace runtime {

Error register_kernels(const Span<const Kernel> kernels) {
  return kernels.empty() ? Error::Ok : Error::InvalidArgument;
}

[[noreturn]] void runtime_abort() {
  std::abort();
}

} // namespace runtime
} // namespace executorch
