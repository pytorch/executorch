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

// Runs encoder, cross_kv and cached decoder methods using parsed command-line
// flags and the engine supplied by the caller. Returns the process exit status.
int run_whisper(const std::function<std::unique_ptr<EngineHost>()>& make_host);

} // namespace ptn
