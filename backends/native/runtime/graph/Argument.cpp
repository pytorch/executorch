// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Argument.h>

#include <stdexcept>
#include <string>

namespace ptn {

void Argument::throw_bad_kind(const char* what) {
  throw std::runtime_error(std::string("Argument::") + what);
}

} // namespace ptn
