/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchUtils.h"

namespace executorch::extension::utils {
using namespace aten;
using namespace runtime;

ScalarType deduceType(NSNumber *number) {
  ET_CHECK(number);
  auto type = [number objCType][0];
  type = (type >= 'A' && type <= 'Z') ? type + ('a' - 'A') : type;
  switch(type) {
    case 'c': return ScalarType::Byte;
    case 's': return ScalarType::Short;
    case 'i': return ScalarType::Int;
    case 'q':
    case 'l': return ScalarType::Long;
    case 'f': return ScalarType::Float;
    case 'd': return ScalarType::Double;
    default: {
      ET_CHECK_MSG(false, "Unsupported type: %c", type);
      return ScalarType::Undefined;
    }
  }
}

} // namespace executorch::extension::utils
