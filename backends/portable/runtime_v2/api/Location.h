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
namespace portable_v2 {

// Process-wide opaque identity, assigned by ProviderRegistry on registration.
// Used for Location, capability queries, registry lookups, debug.
// Never to be hardcoded; never used to index hot-path arrays.
using RuntimeId = uint16_t;
inline constexpr RuntimeId kHost = 0xFFFF;

// Plan-local dense index into Plan::providers / Plan::instances.
// Used only inside Plan/Step/executor. Translated from RuntimeId once at
// route() time. By convention slot 0 is the host (CPU) Provider; CPU is
// required and always present, so kHostIdx == 0 — no sentinel dance.
using RuntimeIndex = uint8_t;
inline constexpr RuntimeIndex kHostIdx = 0;

/**
 * Pure tag describing where a value lives. ~2 bytes; no pointers, no
 * ownership.
 */
class Location {
 public:
  // Default-constructs to host. Convenient for aggregate-initialized
  // structs (InputBinding, OutputBinding) that hold a Location.
  constexpr Location() : id_(kHost) {}

  static Location host() { return Location{kHost}; }
  static Location on(RuntimeId id) { return Location{id}; }

  RuntimeId runtime_id() const { return id_; }
  bool is_host() const { return id_ == kHost; }

  bool operator==(Location other) const { return id_ == other.id_; }
  bool operator!=(Location other) const { return !(*this == other); }

 private:
  explicit constexpr Location(RuntimeId id) : id_(id) {}
  RuntimeId id_;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
