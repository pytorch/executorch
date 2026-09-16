/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/operator_registry.h>

int main() {
  return executorch::runtime::registry_has_op_function("my_ops::mul3.out") ? 0
                                                                           : 1;
}
