/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace evalue_lean_and_aten_test {

// Returns typeid(EValue).name() from a lib built without USE_ATEN_LIB.
const char* lean_evalue_type_name();

} // namespace evalue_lean_and_aten_test
