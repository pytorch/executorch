/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

namespace vkcompute {
namespace vkapi {

class Adapter;

// Test-only device capability overrides. Each set field replaces the
// corresponding Adapter query, simulating a weaker device so fallback paths
// can be covered on capable hardware. Forcing a capability the device
// lacks is undefined behavior and typically fails at pipeline creation.
// Overrides apply to every graph built against the Adapter while set;
// hold a ScopedAdapterCapabilityOverride to bound the scope to one test.
// Not thread-safe: while set, every query on the Adapter observes the
// override, so hold it only around single-test build plus execute with no
// concurrent adapter use.
struct AdapterCapabilityOverrides {
  std::optional<bool> int8_dot_product;
  std::optional<bool> signed_packed4x8_dot;
  std::optional<bool> unsigned_packed4x8_dot;

  static AdapterCapabilityOverrides without_dot_product_support() {
    AdapterCapabilityOverrides overrides;
    overrides.int8_dot_product = false;
    overrides.signed_packed4x8_dot = false;
    overrides.unsigned_packed4x8_dot = false;
    return overrides;
  }
};

class ScopedAdapterCapabilityOverride final {
 public:
  ScopedAdapterCapabilityOverride(
      Adapter* adapter,
      AdapterCapabilityOverrides overrides);
  ~ScopedAdapterCapabilityOverride();
  ScopedAdapterCapabilityOverride(const ScopedAdapterCapabilityOverride&) =
      delete;
  ScopedAdapterCapabilityOverride& operator=(
      const ScopedAdapterCapabilityOverride&) = delete;
  ScopedAdapterCapabilityOverride(ScopedAdapterCapabilityOverride&&) = delete;
  ScopedAdapterCapabilityOverride& operator=(
      ScopedAdapterCapabilityOverride&&) = delete;

 private:
  Adapter* adapter_;
  AdapterCapabilityOverrides previous_;
};

} // namespace vkapi
} // namespace vkcompute
