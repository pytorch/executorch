/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

    .section .rodata.espnn_bundle, "a"
    .balign 128
    .global espnn_bundle_start
espnn_bundle_start:
    .incbin ESPNN_BUNDLE_FILE
    .global espnn_bundle_end
espnn_bundle_end:
    .byte 0

    .section .note.GNU-stack, "", @progbits
