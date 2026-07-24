/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Boot stub for the qemu virt RISC-V baremetal runner: set sp, enable FPU,
// zero .bss, run C++ static ctors via __libc_init_array, jump to main. On
// return, call baremetal_exit so qemu terminates deterministically.

#if __riscv_xlen == 64
#define SX sd
#define XLEN_BYTES 8
#else
#define SX sw
#define XLEN_BYTES 4
#endif

    .section .text.boot, "ax"
    .globl _start
    .type _start, @function
_start:
    la      sp, _stack_top

    // mstatus.FS resets to Off in M-mode, so any FP insn (libstdc++ template
    // code emits fsd/fld) traps. We have no trap vector, so the CPU would
    // loop on the fault. FS=Dirty (0b11 in bits 13-14) keeps the FPU live.
    li      t0, 0x6000
    csrs    mstatus, t0

    la      a0, _bss_start
    la      a1, _bss_end
1:
    bgeu    a0, a1, 2f
    SX      zero, 0(a0)
    addi    a0, a0, XLEN_BYTES
    j       1b
2:
    call    __libc_init_array
    li      a0, 0
    li      a1, 0
    call    main
    call    baremetal_exit
3:
    wfi
    j       3b

    .size _start, .-_start
