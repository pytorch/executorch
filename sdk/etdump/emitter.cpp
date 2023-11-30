/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <cstdint>

#include "executorch/runtime/platform/assert.h"
#include "executorch/sdk/etdump/emitter.h"

namespace torch {
namespace executor {

static int _allocator_fn(
    void* alloc_context,
    flatcc_iovec_t* b,
    size_t request,
    int zero_fill,
    int hint) {
  void* p;
  size_t n;

  struct etdump_static_allocator* state =
      (struct etdump_static_allocator*)alloc_context;

  // This allocator doesn't support freeing memory.
  if (request == 0) {
    if (b->iov_base) {
      b->iov_base = nullptr;
      b->iov_len = 0;
    }
    return 0;
  }

  switch (hint) {
    case flatcc_builder_alloc_ds:
      n = 256;
      break;
    case flatcc_builder_alloc_ht:
      /* Should be exact size, or space size is just wasted. */
      n = request;
      break;
    case flatcc_builder_alloc_fs:
      n = sizeof(__flatcc_builder_frame_t) * 8;
      break;
    case flatcc_builder_alloc_us:
      n = 64;
      break;
    case flatcc_builder_alloc_vd:
      n = 64;
      break;
    default:
      /*
       * We have many small structures - vs stack for tables with few
       * elements, and few offset fields in patch log. No need to
       * overallocate in case of busy small messages.
       */
      n = 32;
      break;
  }

  while (n < request) {
    n *= 2;
  }

  if (b->iov_base != nullptr) {
    if (request > b->iov_len) {
      // We don't support reallocating larger buffers.
      if (((uintptr_t)b->iov_base + b->iov_len) ==
          (uintptr_t)&state->data[state->allocated]) {
        if ((state->allocated + n - b->iov_len) > state->data_size) {
          return -1;
        }
        state->allocated += n - b->iov_len;
      } else {
        if ((state->allocated + n) > state->data_size) {
          return -1;
        }
        memcpy((void*)&state->data[state->allocated], b->iov_base, b->iov_len);
        b->iov_base = &state->data[state->allocated];
        state->allocated += n;
      }
      if (zero_fill) {
        memset((uint8_t*)b->iov_base + b->iov_len, 0, n - b->iov_len);
      }
      b->iov_len = n;
    }

    // Ignore request to resize buffers down.
    return 0;
  }

  if ((state->allocated + n) > state->data_size) {
    return -1;
  }

  p = &state->data[state->allocated];
  state->allocated += n;

  if (zero_fill) {
    memset((void*)p, 0, n);
  }

  b->iov_base = p;
  b->iov_len = n;

  return 0;
}

// This emitter implementation emits to a fixed size buffer and will fail if it
// runs out of room on either end.
static int _emitter_fn(
    void* emit_context,
    const flatcc_iovec_t* iov,
    int iov_count,
    flatbuffers_soffset_t offset,
    size_t len) {
  struct etdump_static_allocator* E =
      (struct etdump_static_allocator*)emit_context;
  uint8_t* p;

  if (offset < 0) {
    if (len > E->front_left) {
      return -1;
    }
    E->front_cursor -= len;
    E->front_left -= len;
    p = E->front_cursor;
  } else {
    ET_CHECK_MSG(
        0, "Moving the back pointer is currently not supported in ETDump.");
  }

  while (iov_count--) {
    memcpy(p, iov->iov_base, iov->iov_len);
    p += iov->iov_len;
    ++iov;
  }

  return 0;
}

/*******************************************************************************
 * Public Functions
 ******************************************************************************/

int etdump_static_allocator_builder_init(
    flatcc_builder_t* builder,
    struct etdump_static_allocator* alloc) {
  ET_CHECK(builder != nullptr);
  ET_CHECK(alloc != nullptr);

  // Ensure data size is multiple of 32 (minimum allocation size).
  ET_CHECK((alloc->data_size & 0x1F) == 0);
  // Ensure out_size is divisable by 2 to ensure front/back sizes are equal for
  // emitter..
  ET_CHECK((alloc->out_size & 0x1) == 0);

  return flatcc_builder_custom_init(
      builder, _emitter_fn, alloc, _allocator_fn, alloc);
}

void etdump_static_allocator_reset(struct etdump_static_allocator* alloc) {
  ET_CHECK(alloc != nullptr);
  alloc->allocated = 0;
  size_t n = alloc->out_size / 2;
  alloc->front_cursor = &alloc->data[alloc->data_size + n];
  alloc->front_left = n;
}

int et_flatcc_custom_init(
    flatcc_builder_t* builder,
    struct etdump_static_allocator* alloc) {
  return flatcc_builder_custom_init(
      builder, _emitter_fn, alloc, _allocator_fn, alloc);
}

} // namespace executor
} // namespace torch
