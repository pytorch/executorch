/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "picolibc_startup.h"

#include <stdint.h>

struct copy_table_entry {
  const uint32_t* source;
  uint32_t* destination;
  uint32_t word_count;
};

struct zero_table_entry {
  uint32_t* destination;
  uint32_t word_count;
};

extern const struct copy_table_entry __copy_table_start__;
extern const struct copy_table_entry __copy_table_end__;
extern const struct zero_table_entry __zero_table_start__;
extern const struct zero_table_entry __zero_table_end__;
extern void _start(void) __attribute__((noreturn));

void executorch_program_start(void) {
  for (uintptr_t entry_address = (uintptr_t)&__copy_table_start__;
       entry_address < (uintptr_t)&__copy_table_end__;
       entry_address += sizeof(struct copy_table_entry)) {
    const struct copy_table_entry* entry =
        (const struct copy_table_entry*)entry_address;
    for (uint32_t i = 0; i < entry->word_count; ++i) {
      entry->destination[i] = entry->source[i];
    }
  }

  for (uintptr_t entry_address = (uintptr_t)&__zero_table_start__;
       entry_address < (uintptr_t)&__zero_table_end__;
       entry_address += sizeof(struct zero_table_entry)) {
    const struct zero_table_entry* entry =
        (const struct zero_table_entry*)entry_address;
    for (uint32_t i = 0; i < entry->word_count; ++i) {
      entry->destination[i] = 0;
    }
  }

  _start();
}
