/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;

inline int get_element_size(ScalarType dtype) {
  if ((dtype == ScalarType::Int) || (dtype == ScalarType::UInt32)) {
    return sizeof(int);
  } else if ((dtype == ScalarType::Short) || (dtype == ScalarType::UInt16)) {
    return sizeof(short);
  } else if ((dtype == ScalarType::Char) || (dtype == ScalarType::Byte)) {
    return sizeof(char);
  }
  return 0;
}
