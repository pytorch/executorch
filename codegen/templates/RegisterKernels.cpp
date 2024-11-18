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
#include "${fn_header}" // Generated Function import headers

namespace torch {
namespace executor {

Error register_all_kernels() {
  Kernel kernels_to_register[] = {
      ${unboxed_kernels} // Generated kernels
  };
  Error success_with_kernel_reg =
      ::executorch::runtime::register_kernels({kernels_to_register});
  if (success_with_kernel_reg != Error::Ok) {
    ET_LOG(Error, "Failed register all kernels");
    return success_with_kernel_reg;
  }
  return Error::Ok;
}

} // namespace executor
} // namespace torch
