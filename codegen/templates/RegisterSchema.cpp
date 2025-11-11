/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ${generated_comment}
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/library.h>

namespace at {
TORCH_LIBRARY_FRAGMENT(aten, m) {
  ${aten_schema_registrations};
}
$schema_registrations
} // namespace at
