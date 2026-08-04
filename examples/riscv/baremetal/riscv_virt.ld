/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* qemu-system-riscv{32,64} -machine virt -bios none -kernel: the virt board's
 * reset stub at 0x1000 jumps to DRAM base 0x80000000, so _start has to live
 * there. RAM size matches the qemu `-m 512M` we pass from run.sh — the
 * embedded .bpte in .rodata can be tens of MB for mv2 / llama2 / yolo26. */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 512M
}

SECTIONS
{
    .text 0x80000000 :
    {
        KEEP(*(.text.boot))
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(8)
    {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    /* C++ global ctors. start.S calls picolibc's __libc_init_array, which
     * walks symbols __bothinit_array_start..__bothinit_array_end (preinit +
     * init combined). The stock newlib names (__init_array_start/end) are
     * defined too for portability, but it's the "both" pair picolibc reads. */
    .bothinit_array : ALIGN(8)
    {
        PROVIDE_HIDDEN(__bothinit_array_start = .);
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP(*(.init_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o) .ctors))
        PROVIDE_HIDDEN(__init_array_end = .);
        PROVIDE_HIDDEN(__bothinit_array_end = .);
    } > RAM
    .fini_array : ALIGN(8)
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
        KEEP(*(.fini_array EXCLUDE_FILE(*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o) .dtors))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > RAM

    .data : ALIGN(8)
    {
        *(.data .data.*)
        *(.sdata .sdata.*)
    } > RAM

    .bss : ALIGN(8)
    {
        _bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(8);
        _bss_end = .;
    } > RAM

    /* 2 MiB stack at the high end of RAM; grows downward. picolibc's sbrk
     * looks up __heap_start / __heap_end (double-underscore). */
    . = ALIGN(16);
    PROVIDE(__heap_start = .);
    . = ORIGIN(RAM) + LENGTH(RAM) - 2M;
    PROVIDE(__heap_end = .);
    . = . + 2M;
    _stack_top = .;

    /DISCARD/ : { *(.note.* .comment .eh_frame .riscv.attributes) }
}
