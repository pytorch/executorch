/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <manual_ops_1_lib/RegisterKernels.h>
#include <manual_ops_2_lib/RegisterKernels.h>
#ifndef INSTALLED_CONSUMER
#include <manual_ops_overlap_lib/RegisterKernels.h>
#endif

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstring>

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  if (executorch::runtime::registry_has_op_function("my_ops::mul3.out") ||
      executorch::runtime::registry_has_op_function("my_ops::mul4.out")) {
    return 5;
  }
  if (torch::executor::register_manual_ops_1_lib_kernels() !=
      executorch::runtime::Error::Ok) {
    return 1;
  }
  if (torch::executor::register_manual_ops_2_lib_kernels() !=
      executorch::runtime::Error::Ok) {
    return 2;
  }
  if (!executorch::runtime::registry_has_op_function("my_ops::mul3.out")) {
    return 3;
  }
  if (!executorch::runtime::registry_has_op_function("my_ops::mul4.out")) {
    return 4;
  }
#ifndef INSTALLED_CONSUMER
  if (argc == 2 && std::strcmp(argv[1], "--overlap") == 0) {
    torch::executor::register_manual_ops_overlap_lib_kernels();
    return 6;
  }
#else
  (void)argc;
  (void)argv;
#endif
  return 0;
}
