/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/testing_util/error_matchers.h>

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace runtime {

// This needs to be defined in the SAME namespace that defines Error.
// C++'s look-up rules rely on that.
void PrintTo(const Error& error, std::ostream* os) {
  *os << ::executorch::runtime::to_string(error);
}

} // namespace runtime
} // namespace executorch
