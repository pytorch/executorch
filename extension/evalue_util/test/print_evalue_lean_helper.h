/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>

namespace print_evalue_lean_and_aten_test {

// Prints a 7-item double list EValue from a lib built without USE_ATEN_LIB.
void print_lean_double_list(std::ostream& os);

} // namespace print_evalue_lean_and_aten_test
