/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace backends {
namespace native {

// Process-wide opaque identity, assigned by RuntimeRegistry on registration.
// Used for capability queries, registry lookups, debug.
// Never to be hardcoded; never used to index hot-path arrays.
using RuntimeId = uint16_t;
inline constexpr RuntimeId kHost = 0xFFFF;

// Plan-local dense index into Plan::providers / Plan::instances.
// Used only inside Plan/Step/executor. Translated from RuntimeId once at
// route() time. By convention slot 0 is the host (CPU) Runtime; CPU is
// required and always present, so kHostIdx == 0 — no sentinel dance.
using RuntimeIndex = uint8_t;
inline constexpr RuntimeIndex kHostIdx = 0;

} // namespace native
} // namespace backends
} // namespace executorch
