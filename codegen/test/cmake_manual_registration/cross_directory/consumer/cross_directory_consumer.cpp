/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/cross_dir_ops/RegisterKernels.h>

int main() {
  return torch::executor::register_stable_cross_dir_kernels() ==
          executorch::runtime::Error::Ok
      ? 0
      : 1;
}
