// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

namespace ptn {

// The GgmlEngineOptions that the command-line flags every ggml runner shares
// select, so that each runner accepts them with the same meaning and help.
GgmlEngineOptions ggml_engine_options_from_flags();

} // namespace ptn
