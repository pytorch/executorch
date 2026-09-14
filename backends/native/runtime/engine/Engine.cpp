// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

// Out of line so each vtable is emitted here rather than in every translation
// unit that includes the header.
EngineExecutable::~EngineExecutable() = default;

EngineContext::~EngineContext() = default;

} // namespace ptn
