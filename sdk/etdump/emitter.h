/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>
#include <stdlib.h>

#include <flatcc/flatcc_builder.h>

#pragma once

namespace torch {
namespace executor {

struct etdump_static_allocator {
  etdump_static_allocator(
      uint8_t* buffer,
      size_t total_buf_size,
      size_t alloc_buf_size)
      : data{buffer},
        data_size{alloc_buf_size},
        allocated{0},
        out_size{total_buf_size - alloc_buf_size},
        front_cursor{&buffer[alloc_buf_size]},
        front_left{out_size / 2} {}
  // Pointer to backing buffer to allocate from.
  uint8_t* data{nullptr};

  // Size of backing buffer.
  size_t data_size{0};

  // Current allocation offset.
  size_t allocated{0};

  // Size of build buffer.
  size_t out_size{0};

  // Pointer to front of build buffer.
  uint8_t* front_cursor{nullptr};

  // Bytes left in front of front_cursor.
  size_t front_left{0};
};

int et_flatcc_custom_init(
    flatcc_builder_t* builder,
    struct etdump_static_allocator* alloc);

} // namespace executor
} // namespace torch
