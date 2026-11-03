/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

    .section .rodata.smoke_pte, "a"
    .balign 16
    .global smoke_pte_start
smoke_pte_start:
    .incbin ET_SMOKE_PTE
    .global smoke_pte_end
smoke_pte_end:
    .byte 0

    .section .note.GNU-stack, "", @progbits
