// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn::testing {

// Defined once per runner binary; ptn_runner_main.cpp drives the host it
// returns, so every engine runs a PTN through the same harness.
std::unique_ptr<EngineHost> make_engine_host();

} // namespace ptn::testing
