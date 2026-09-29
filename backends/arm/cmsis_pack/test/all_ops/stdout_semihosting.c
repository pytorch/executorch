/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * CMSIS-Compiler STDOUT/STDERR:Custom backend. The CMSIS-Compiler CORE
 * component provides the toolchain-specific low-level retarget (newlib _write
 * for GCC, the Arm C library retarget for AC6, picolibc FILE streams for AtFE)
 * and routes every character here. We emit it over Arm semihosting (BKPT 0xAB,
 * SYS_WRITEC) so printf / assert reach the FVP console with no compiler-specific
 * syscall headers. STDERR is needed because picolibc's assert prints to stderr.
 */

#include "retarget_stdout.h"
#include "retarget_stderr.h"

static int semihosting_putc(int ch) {
  char c = (char)ch;
  register int r0 __asm__("r0") = 0x03; /* SYS_WRITEC */
  register void *r1 __asm__("r1") = &c;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return ch;
}

int stdout_putchar(int ch) { return semihosting_putc(ch); }
int stderr_putchar(int ch) { return semihosting_putc(ch); }
