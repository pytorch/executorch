/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/Types.h>

namespace vkcompute {

#define PRINT_CASE(name) \
  case TypeTag::name:    \
    out << #name;        \
    break;

std::ostream& operator<<(std::ostream& out, const TypeTag& tag) {
  switch (tag) {
    PRINT_CASE(NONE)
    PRINT_CASE(INT)
    PRINT_CASE(DOUBLE)
    PRINT_CASE(BOOL)
    PRINT_CASE(TENSOR)
    PRINT_CASE(STAGING)
    PRINT_CASE(TENSORREF)
    PRINT_CASE(INTLIST)
    PRINT_CASE(DOUBLELIST)
    PRINT_CASE(BOOLLIST)
    PRINT_CASE(VALUELIST)
    PRINT_CASE(STRING)
    PRINT_CASE(SYMINT)
  }
  return out;
}

} // namespace vkcompute
