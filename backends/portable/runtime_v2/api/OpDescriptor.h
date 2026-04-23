/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string_view>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * What the Provider sees for a single op when asked can_run().
 *
 * Currently carries only the op name. Per-value descriptors (dtype,
 * shape, dynamism, etc.) and capability/cost return types are NOT yet
 * here — they're additive when multi-provider routing actually needs
 * them, and adding fields to this struct is non-breaking.
 */
struct OpDescriptor {
  std::string_view name;  // e.g. "aten.add.Tensor"
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
