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
namespace native {

/**
 * What the Runtime sees for a single op when asked can_run().
 *
 * Currently carries only the op name. Per-value descriptors (dtype,
 * shape, dynamism, etc.) and capability/cost return types are NOT yet
 * here — they're additive when multi-provider routing actually needs
 * them, and adding fields to this struct is non-breaking.
 */
/**
 * Op identity passed to Runtime::can_run() for capability queries.
 *
 * `name` is the base op name (e.g., "aten::add"), without overload
 * disambiguation. This is best-effort capability — a Runtime may say
 * yes for the base name even if a particular overload is unsupported;
 * actual overload-specific dispatch failure is detected at execute
 * time as Error::NotSupported.
 *
 * Per-value descriptors (dtype, shape, dynamism, etc.) and
 * capability/cost return types are NOT yet here — they're additive
 * when multi-provider routing actually needs them, and adding fields
 * to this struct is non-breaking.
 */
struct OpDescriptor {
  std::string_view name; // e.g. "aten::add" (base name, no overload)
};

} // namespace native
} // namespace backends
} // namespace executorch
