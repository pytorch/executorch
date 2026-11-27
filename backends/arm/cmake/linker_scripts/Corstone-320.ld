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
 * https://developer.arm.com/Processors/Corstone-320
 */

/* default value - '1', for DRAM */
#ifndef ETHOSU_MODEL
#define ETHOSU_MODEL  1
#endif

/* default value - '1', for DRAM */
#ifndef ETHOSU_ARENA
#define ETHOSU_ARENA  1
#endif

#ifndef STACK_SIZE
#define STACK_SIZE 0x8000
#endif

#ifndef HEAP_SIZE
#define HEAP_SIZE 0x10000
#endif

__STACK_SIZE = STACK_SIZE;
__HEAP_SIZE  = HEAP_SIZE;

MEMORY
{
  ITCM  (rwx) : ORIGIN = 0x10000000, LENGTH = 0x00008000
  BROM  (rx)  : ORIGIN = 0x11000000, LENGTH = 0x00020000
  BRAM  (rwx) : ORIGIN = 0x12000000, LENGTH = 0x00200000
  DTCM  (rw)  : ORIGIN = 0x30000000, LENGTH = 0x00008000
  SRAM  (rw)  : ORIGIN = 0x31000000, LENGTH = 0x00400000
  QSPI  (rw)  : ORIGIN = 0x38000000, LENGTH = 0x00800000
  DDR   (rw)  : ORIGIN = 0x70000000, LENGTH = 0x10000000
}

PHDRS
{
    rom_boot PT_LOAD;
    rom_exec PT_LOAD;
    rom_sram PT_LOAD;
    rom_dram PT_LOAD;
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
  .text.vectors :
  {
    KEEP(*(.vectors))
  } > BROM :rom_boot

  /*
  /* Vector table relocation to read write memory
   * Alignment requirement from up to 496 interrupts, rounded to the closest
   * power of two equals 512 (words), thus 2048 bytes.
   */
#ifdef __clang__
  .data.vtable_rw (NOLOAD):
#else
  .data.vtable_rw (COPY):
#endif
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
    *(.got)
    *(.got.*)
    *(.rodata*)

    . = ALIGN(4);
#ifndef __clang__
    __data_end__ = .;
#endif
  } > BRAM :rom_exec

#ifdef __clang__
  .tdata :
  {
    __tls_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    . = ALIGN(4);
    __data_end__ = .;
  } > BRAM :rom_exec :tls

  .tbss (NOLOAD) :
  {
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    . = ALIGN(4);
    __tls_end = .;
  } > BRAM :null :tls
#endif

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

#ifdef __clang__
    LONG (ADDR(.tbss))
    LONG (SIZEOF(.tbss) / 4)
#endif

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
  } > SRAM :rom_sram

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
    . = ALIGN(4);
  } > SRAM :null

  .ddr :
  {
    . = ALIGN(4);
    *(input_data_sec)
    *(expected_output_data_sec)
    *(output_data_sec)

    *(ethosu_core_in_queue ethosu_core_out_queue)

    /* Place data for scatter loading here */
    __etext = .;
  } > DDR :rom_dram
  .asan_shadow (NOLOAD) :
  {
    . = ALIGN(16);
    __asan_shadow_start = .;
    KEEP(*(.asan_shadow))
    __asan_shadow_end = .;
  } > DDR :null

  .ddr.bss (NOLOAD) :
  {
    . = ALIGN(16);
    *(.bss.input_file_allocator_sec)
    *(.bss.method_allocator_sec)
    *(input_file_allocator_sec)
    *(method_allocator_sec)
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
  } > BRAM :null

  .heap (ORIGIN(BRAM) + LENGTH(BRAM) - __HEAP_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > BRAM :null

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

  /* Check if stack exceeds DTCM limit */
  ASSERT(LENGTH(DTCM) >= __STACK_SIZE, "region DTCM overflowed with stack")
}
