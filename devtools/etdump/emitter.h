/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#include <executorch/devtools/etdump/etdump_flatcc.h>

typedef struct flatcc_builder flatcc_builder_t;

namespace executorch {
namespace etdump {
namespace internal {

int etdump_flatcc_custom_init(
    flatcc_builder_t* builder,
    internal::ETDumpStaticAllocator* alloc);

} // namespace internal
} // namespace etdump
} // namespace executorch
