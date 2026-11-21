/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noreturn)) void executorch_program_start(void);

#ifdef __cplusplus
}
#endif

#define __PROGRAM_START executorch_program_start
