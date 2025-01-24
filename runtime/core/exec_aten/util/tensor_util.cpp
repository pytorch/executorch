#include "tensor_util.h"

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/runtime/platform/assert.h>

namespace executorch::runtime {
/**
 * Shared implementation for tensor_util.h, may only contain code that
 * works whether or not ATen mode is active.
 */
void tensor_shape_to_c_string(
    char out[kTensorShapeStringSizeLimit],
    executorch::aten::ArrayRef<executorch::aten::SizesType> shape) {
  char* p = out;
  *p++ = '(';
  for (const auto elem : shape) {
    if (elem < 0 || elem > kMaximumPrintableTensorShapeElement) {
      strcpy(p, "ERR, ");
      p += strlen("ERR, ");
    } else {
      // snprintf returns characters *except* the NUL terminator, which is what
      // we want.
      p += snprintf(
          p,
          kTensorShapeStringSizeLimit - (p - out),
          "%" PRIu32 ", ",
          static_cast<uint32_t>(elem));
    }
  }
  *(p - 2) = ')';
  *(p - 1) = '\0';
}

} // namespace executorch::runtime
