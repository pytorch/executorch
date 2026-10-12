/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/test/evalue_lean_type_name.h>

#include <typeinfo>

#include <executorch/runtime/core/evalue.h>

namespace evalue_lean_and_aten_test {

const char* lean_evalue_type_name() {
  return typeid(executorch::runtime::EValue).name();
}

} // namespace evalue_lean_and_aten_test
