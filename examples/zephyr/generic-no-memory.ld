/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Link-only size-test script; this is not a runnable Zephyr memory map. */

ENTRY(main)

SECTIONS
{
  . = 0x0;
  .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) }
  .ARM.exidx : {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
  }
  .preinit_array : { KEEP(*(.preinit_array*)) }
  .init_array : { KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*))) KEEP(*(.init_array)) }
  .fini_array : { KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*))) KEEP(*(.fini_array)) }
  .data : { *(.data*) }
  .bss : { *(.bss*) *(COMMON) }
}
