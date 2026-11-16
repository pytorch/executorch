/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
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
 * https://developer.arm.com/Processors/Corstone-300
 */

#ifndef ETHOSU_MODEL
  /* default value - '1', for DRAM */
  #define ETHOSU_MODEL  1
#endif

#ifndef ETHOSU_ARENA
  /* default value - '1', for DRAM */
  #define ETHOSU_ARENA  1
#endif

#ifndef STACK_SIZE
  #define STACK_SIZE 0x00008000
#endif

#ifndef HEAP_SIZE
  #define HEAP_SIZE 0x00008000
#endif

__STACK_SIZE = STACK_SIZE;
__HEAP_SIZE  = HEAP_SIZE;

MEMORY
{
  ITCM  (rx)  : ORIGIN = 0x10000000, LENGTH = 0x00080000
  BRAM  (rw)  : ORIGIN = 0x11000000, LENGTH = 0x00100000
  DTCM  (rw)  : ORIGIN = 0x30000000, LENGTH = 0x00080000
  SRAM  (rw)  : ORIGIN = 0x31000000, LENGTH = 0x00200000
  QSPI  (rw)  : ORIGIN = 0x38000000, LENGTH = 0x00800000
  DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x60000000
}

PHDRS
{
    rom_exec PT_LOAD;
    rom_data PT_LOAD;
    rom_dram PT_LOAD;
    rom_sram PT_LOAD;
    rom_rodata PT_LOAD;
    rom_model PT_LOAD;
#ifdef __clang__
    tls       PT_TLS;
#endif
    null     PT_NULL;
}

/* Linker script to place sections and symbol values. Should be used together
 * with other linker script that defines memory regions ITCM and RAM.
 * It references following symbols, which must be defined in code:
 *   Reset_Handler : Entry of reset handler
 *
 * It defines following symbols, which code can use without definition:
 *   __exidx_start
 *   __exidx_end
 *   __copy_table_start__
 *   __copy_table_end__
 *   __zero_table_start__
 *   __zero_table_end__
 *   __etext
 *   __data_start__
 *   __preinit_array_start
 *   __preinit_array_end
 *   __init_array_start
 *   __init_array_end
 *   __fini_array_start
 *   __fini_array_end
 *   __data_end__
 *   __bss_start__
 *   __bss_end__
 *   __end__
 *   end
 *   __HeapLimit
 *   __StackLimit
 *   __StackTop
 *   __stack
 */

ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    _vectors = .;
    KEEP(*(.vectors))
    *(EXCLUDE_FILE(
                  *op_*.cpp.obj
                   )
    .text*)
    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    KEEP(*(.eh_frame*))
    *(.got)
    *(.got.*)
  } > ITCM :rom_exec

  /*
   * SG veneers:
   * All SG veneers are placed in the special output section .gnu.sgstubs. Its start address
   * must be set, either with the command line option '--section-start' or in a linker script,
   * to indicate where to place these veneers in memory.
   */
/*
  .gnu.sgstubs :
  {
    . = ALIGN(32);
  } > ITCM :rom_exec
*/
  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM :rom_exec

  .ARM.exidx :
  {
  __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  __exidx_end = .;
  } > ITCM :rom_exec

#ifdef __clang__
  .copy.table :
#else
  .copy.table (READONLY) :
#endif
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (__etext)
    LONG (__data_start__)
    LONG ((__data_end__ - __data_start__) / 4)

    LONG (__eddr_data)
    LONG (__sram_data_start__)
    LONG ((__sram_data_end__ - __sram_data_start__) / 4)

    LONG (LOADADDR(.rodata))
    LONG (__rodata_start__)
    LONG ((__rodata_end__ - __rodata_start__) / 4)

    __copy_table_end__ = .;
  } > ITCM :rom_exec

#ifdef __clang__
  .zero.table :
#else
  .zero.table (READONLY) :
#endif
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
#ifdef __clang__
    LONG (ADDR(.tbss))
    LONG (SIZEOF(.tbss) / 4)
#endif

    LONG (__bss_start__)
    LONG ((__bss_end__ - __bss_start__) / 4)
    __zero_table_end__ = .;

  /**
   * Location counter can end up 2byte aligned with narrow Thumb code but
   * __etext is assumed by startup code to be the LMA of a section in DTCM
   * which must be 4byte aligned
   */
  __etext = ALIGN (4);

  } > ITCM :rom_exec

  .data : AT(__etext)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data */
#ifdef __clang__
    __bothinit_array_start = .;
#endif
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
#ifdef __clang__
    __bothinit_array_end = .;
#endif

    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    /* All data end */
#ifndef __clang__
    __data_end__ = .;
#endif
  } > DTCM :rom_data

#ifdef __clang__
  .tdata : AT(LOADADDR(.data) + SIZEOF(.data))
  {
    __tls_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    . = ALIGN(4);
    __data_end__ = .;
  } > DTCM :rom_data :tls

  .tbss (NOLOAD) :
  {
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    . = ALIGN(4);
    __tls_end = .;
  } > DTCM :null :tls
#endif

#if (ETHOSU_MODEL == 0)
  .sram.rodata :
  {
    . = ALIGN(16);
    *(network_model_sec)
  } > SRAM :rom_model
#else
  .ddr.rodata :
  {
    . = ALIGN(16);
    *(network_model_sec)
  } > DDR :rom_model
#endif

  .sram.bss (NOLOAD) :
  {
#if (ETHOSU_ARENA == 0)
    . = ALIGN(32);
    *(.bss.tensor_arena)
#endif
    . = ALIGN(16);
    *(.fast_memory)
    . = ALIGN(16);
    *(.bss.ethosu_scratch);
  } > SRAM :null

  .ddr :
  {
    . = ALIGN(4);
    *(input_data_sec)
    . = ALIGN(16);
    * (expected_output_data_sec)
    . = ALIGN(16);
    * (sec_command_stream, sec_weight_data, sec_input_data)
    * (ethosu_core_in_queue)
    * (ethosu_core_out_queue)
    . = ALIGN(4);
  } > DDR :rom_dram
  .asan_shadow (NOLOAD) :
  {
    . = ALIGN(16);
    __asan_shadow_start = .;
    KEEP(*(.asan_shadow))
    __asan_shadow_end = .;
  } > DDR :null

  __eddr_data = ALIGN(4);
  .sram.data :
  {
    __sram_data_start__ = .;
    *(.sram.data)
    . = ALIGN(4);
    *op_*.cpp.obj (*.text*)
    . = ALIGN(4);
    __sram_data_end__ = .;
  } > BRAM AT >DDR :rom_sram

  .rodata :
  {
    __rodata_start__ = .;
    *(.rodata)
    *(.rodata.*)
    . = ALIGN(4);
    __rodata_end__ = .;
  } > DTCM AT >DDR :rom_rodata

  .ddr.bss (NOLOAD) :
  {
    . = ALIGN(16);
    *(.bss.input_file_allocator_sec)
    *(.bss.method_allocator_sec)
#if (ETHOSU_ARENA == 1)
    . = ALIGN(32);
    *(.bss.tensor_arena)
#endif
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
  } > DTCM :null

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM :null

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM :null
  PROVIDE(__stack = __StackTop);

#ifdef __clang__
  PROVIDE(__data_start = __data_start__);
  PROVIDE(__data_source = LOADADDR(.data));
  PROVIDE(__data_size = __data_end__ - __data_start__);
  PROVIDE(__bss_start = __bss_start__);
  PROVIDE(__bss_size = __bss_end__ - __bss_start__);
  PROVIDE(__tls_align = MAX(ALIGNOF(.tdata), ALIGNOF(.tbss)));
  PROVIDE(__tls_base = __tls_start);
  PROVIDE(__arm32_tls_tcb_offset = MAX(8, __tls_align));
  PROVIDE(__heap_start = __end__);
  PROVIDE(__heap_end = __HeapLimit);
  PROVIDE(__stack_limit = __StackLimit);
#endif

  __RAM_segment_used_end__ = .;

  /* Check if data + heap + stack exceeds DTCM limit */
  ASSERT(__StackLimit >= __HeapLimit, "region DTCM overflowed with stack")
}
