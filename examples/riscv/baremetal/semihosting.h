/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>

namespace executorch {
namespace riscv {
namespace baremetal {

// The RISC-V semihosting trigger is a fixed three-insn sequence (slli/ebreak/
// srai of x0) so qemu can distinguish it from a normal ecall. Op number in
// a0, arg pointer in a1, return value back in a0.
inline long semihost_call(long op, const void* arg) {
  register long a0 asm("a0") = op;
  register long a1 asm("a1") = (long)arg;
  asm volatile(
      ".option push\n\t"
      ".option norvc\n\t"
      "slli x0, x0, 0x1f\n\t"
      "ebreak\n\t"
      "srai x0, x0, 0x7\n\t"
      ".option pop"
      : "+r"(a0)
      : "r"(a1)
      : "memory");
  return a0;
}

constexpr long SYS_WRITE0 = 0x04;
constexpr long SYS_EXIT_EXTENDED = 0x20;

inline void semihost_write0(const char* s) {
  semihost_call(SYS_WRITE0, s);
}

[[noreturn]] inline void semihost_exit(int status) {
  // ADP_Stopped_ApplicationExit (0x20026) + status, per the semihosting spec.
  long block[2] = {0x20026, (long)status};
  semihost_call(SYS_EXIT_EXTENDED, block);
  __builtin_trap();
}

} // namespace baremetal
} // namespace riscv
} // namespace executorch
