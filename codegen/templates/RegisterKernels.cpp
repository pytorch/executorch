/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ${generated_comment}
// This implements register_all_kernels() API that is declared in
// RegisterKernels.h
#include "RegisterKernels.h"
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include "${fn_header}" // Generated Function import headers

namespace torch {
namespace executor {

#if USE_LIB_NAME_IN_REGISTER
Error register_kernels_${lib_name}() {
#else
Error register_all_kernels() {
#endif

  Kernel kernels_to_register[] = {
      ${unboxed_kernels} // Generated kernels
  };
  Error success_with_kernel_reg =
      ::executorch::runtime::register_kernels({kernels_to_register});
  if (success_with_kernel_reg != Error::Ok) {
    #if USE_LIB_NAME_IN_REGISTER
      ET_LOG(Error, "Failed to register %zu kernels for %s (from %s)",
            sizeof(kernels_to_register) / sizeof(Kernel),
            "${lib_name}",
            __FILE__);
    #else
      ET_LOG(Error, "Failed to register %zu kernels (from %s)",
            sizeof(kernels_to_register) / sizeof(Kernel),
            __FILE__);
    #endif
    return success_with_kernel_reg;
}
  return Error::Ok;
}

} // namespace executor
} // namespace torch
