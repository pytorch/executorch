/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>
#include <stdlib.h>

#include <executorch/sdk/etdump/etdump_flatcc.h>
#include <flatcc/flatcc_builder.h>

#pragma once

namespace torch {
namespace executor {

int et_flatcc_custom_init(
    flatcc_builder_t* builder,
    struct etdump_static_allocator* alloc);

int etdump_static_allocator_builder_init(
    flatcc_builder_t* builder,
    struct etdump_static_allocator* alloc);

void etdump_static_allocator_reset(struct etdump_static_allocator* alloc);

} // namespace executor
} // namespace torch
