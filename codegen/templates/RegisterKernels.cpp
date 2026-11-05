/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ${generated_comment}
// This implements ${manual_registration_function_name}() API that is declared
// in RegisterKernels.h
#include "RegisterKernels.h"
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include "${fn_header}" // Generated Function import headers

namespace torch {
namespace executor {

Error ${manual_registration_function_name}() {
  ${kernel_array_begin}
      ${unboxed_kernels}
  ${kernel_array_end}
  Span<const Kernel> kernel_span(
      ${kernel_data}, ${kernel_count});
  Error success_with_kernel_reg =
      ::executorch::runtime::register_kernels(kernel_span);
  if (success_with_kernel_reg != Error::Ok) {
    ET_LOG(Error, "Failed to register kernels");
    return success_with_kernel_reg;
  }
  return Error::Ok;
}

} // namespace executor
} // namespace torch
