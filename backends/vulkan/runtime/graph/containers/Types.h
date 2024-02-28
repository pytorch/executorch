/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <cstdint>
#include <ostream>

namespace at {
namespace native {
namespace vulkan {

/*
 * This class is modelled after c10::IValue; however, it is simplified and does
 * not support as many types. However, the core design is the same; it is a
 * tagged union over the types supported by the Vulkan Graph type.
 */
enum class TypeTag : uint32_t {
  NONE,
  TENSOR,
  STAGING,
  TENSORREF,
  INT,
  DOUBLE,
  BOOL,
};

std::ostream& operator<<(std::ostream& out, const TypeTag& tag);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
