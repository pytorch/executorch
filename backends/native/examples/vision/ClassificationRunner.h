// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <memory>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

// Reads the parsed command-line flags and runs a single-input, single-output
// FP32 classifier. Exit codes: 0 success, 1 invalid arguments, 2 execution
// error, 3 reference mismatch.
int run_classification(
    const std::function<std::unique_ptr<EngineHost>()>& make_host);

} // namespace ptn
