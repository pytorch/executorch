/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

    .section .rodata.camera_model, "a"
    .balign 128
    .global camera_model_start
camera_model_start:
    .incbin CAMERA_MODEL_FILE
    .global camera_model_end
camera_model_end:
    .byte 0

    .section .note.GNU-stack, "", @progbits
