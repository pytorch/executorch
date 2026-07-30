/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Shim <sys/types.h> for Arm Compiler 6 (AC6).
 *
 * runtime/platform/compiler.h includes <sys/types.h> on every non-MSVC
 * compiler to pick up ssize_t. AC6 targeting arm-arm-none-eabi is a
 * freestanding toolchain and ships no such header in its sysroot, so the
 * include fails. GCC (newlib) and ATfE / arm-llvm-embedded (picolibc)
 * both ship a real <sys/types.h>, so for them this shim forwards via
 * #include_next; for AC6 it provides the minimal subset ExecuTorch needs.
 *
 * This file is part of the PyTorch::ExecuTorch CMSIS Pack and is added
 * to the include search path only when the AC6 condition is active
 * (see the Runtime component in PyTorch.ExecuTorch.pdsc).
 */
#ifndef PYTORCH_EXECUTORCH_AC6_SYS_TYPES_H
#define PYTORCH_EXECUTORCH_AC6_SYS_TYPES_H

#if !defined(__ARMCC_VERSION)
/* GCC newlib / ATfE picolibc — forward to the real header. */
#include_next <sys/types.h>
#else
/* AC6 — provide the minimal set ExecuTorch needs. */
#include <stddef.h>
#include <stdint.h>

#if !defined(_SSIZE_T_DEFINED) && !defined(__ssize_t_defined)
#define _SSIZE_T_DEFINED
#define __ssize_t_defined
#ifdef __cplusplus
using ssize_t = ptrdiff_t;
#else
typedef ptrdiff_t ssize_t;
#endif
#endif

#endif /* !__ARMCC_VERSION */

#endif /* PYTORCH_EXECUTORCH_AC6_SYS_TYPES_H */
