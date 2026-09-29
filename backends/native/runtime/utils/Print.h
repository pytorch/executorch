// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/graph/utils/Print.h>

namespace ptn {

// The method layer of the IR printer, split out only because Method sits above
// the graph/ package. Same contract: debug-only, for humans, not parsed back.

// Multi-line: the method name, its graph, then the binding tables.
std::string to_string(const Method& method);

} // namespace ptn
