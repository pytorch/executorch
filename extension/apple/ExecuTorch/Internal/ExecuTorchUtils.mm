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
  auto type = [number objCType][0];
  type = (type >= 'A' && type <= 'Z') ? type + ('a' - 'A') : type;
  if (type == 'c') {
    return ScalarType::Byte;
  } else if (type == 's') {
    return ScalarType::Short;
  } else if (type == 'i') {
    return ScalarType::Int;
  } else if (type == 'q' || type == 'l') {
    return ScalarType::Long;
  } else if (type == 'f') {
    return ScalarType::Float;
  } else if (type == 'd') {
    return ScalarType::Double;
  }
  ET_CHECK_MSG(false, "Unsupported type: %c", type);
  return ScalarType::Undefined;
}

} // namespace executorch::extension::utils
