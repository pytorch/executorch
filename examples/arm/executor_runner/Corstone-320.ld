/*
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

 /*
 * This is a simplified linkerscript for the Corstone-300 memory system.
 * This example has been modified to place certain sections in specific memory.
 * Please refer to the Corstone SSE-300 Technical Reference Manual for
 * further information.
 *
 * https://developer.arm.com/Processors/Corstone-320
 */

__STACK_SIZE = 0x8000;
__HEAP_SIZE = 0x10000;
MEMORY
{
  ITCM (rwx) : ORIGIN = 0x10000000, LENGTH = 0x00008000
  BROM (rx) : ORIGIN = 0x11000000, LENGTH = 0x00020000
  BRAM (rwx) : ORIGIN = 0x12000000, LENGTH = 0x00200000
  DTCM (rw) : ORIGIN = 0x30000000, LENGTH = 0x00008000
  SRAM (rw) : ORIGIN = 0x31000000, LENGTH = 0x00400000
  QSPI (rw) : ORIGIN = 0x38000000, LENGTH = 0x00800000
  DDR (rw) : ORIGIN = 0x70000000, LENGTH = 0x10000000
}
PHDRS
{
    rom_boot PT_LOAD;
    rom_exec PT_LOAD;
    rom_dram PT_LOAD;
    null PT_NULL;
}
ENTRY(Reset_Handler)
SECTIONS
{
  .text.vectors :
  {
    KEEP(*(.vectors))
  } > BROM :rom_boot
  .data.vtable_rw (COPY):
  {
    . = ALIGN(0x800);
    KEEP(*(.vtable_rw))
  } > ITCM :null
  .text :
  {
    *crt* (.text*)
    *startup_ARMCM85.c.obj (.text*)
    *system_ARMCM85.c.obj (.text*)
    *target.cpp.obj (.text*)
    KEEP(*(.init))
    KEEP(*(.fini))
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)
    *(.text*)
    KEEP(*(.eh_frame*))
  } > BRAM :rom_exec
  .data :
  {
    . = ALIGN(4);
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)
    *(.rodata*)
    . = ALIGN(4);
    __data_end__ = .;
  } > BRAM :rom_exec
  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > BRAM :rom_exec
  .ARM.exidx :
  {
  __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  __exidx_end = .;
  } > BRAM :rom_exec
  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (LOADADDR(.sram))
    LONG (ADDR(.sram))
    LONG (SIZEOF(.sram) / 4)
    __copy_table_end__ = .;
  } > BRAM :rom_exec
  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    LONG (ADDR(.bss))
    LONG (SIZEOF(.bss) / 4)
    LONG (ADDR(.sram.bss))
    LONG (SIZEOF(.sram.bss) / 4)
    __zero_table_end__ = .;
  } > BRAM :rom_exec
  .sram : AT(__etext)
  {
    . = ALIGN(16);
    *(.sram.data)
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);
    KEEP(*(.jcr*))
  } > SRAM :rom_dram
  .sram.bss :
  {
    . = ALIGN(16);
    *(.bss.ethosu_scratch);
  } > SRAM :null
  .ddr :
  {
    . = ALIGN(32);
    *(.bss.tensor_arena)
    . = ALIGN(16);
    *(network_model_sec)
    *(expected_output_data_sec)
    *(output_data_sec)
    *(ethosu_core_in_queue ethosu_core_out_queue)
    __etext = .;
  } > DDR :rom_dram
  .ddr_noload (NOLOAD) :
  {
    . = ALIGN(16);
    *(input_data_sec)
  } > DDR :null
  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > BRAM :null
  .heap (ORIGIN(BRAM) + LENGTH(BRAM) - __HEAP_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > BRAM :null
  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM :null
  PROVIDE(__stack = __StackTop);
  ASSERT(LENGTH(DTCM) >= __STACK_SIZE, "region DTCM overflowed with stack")
}
