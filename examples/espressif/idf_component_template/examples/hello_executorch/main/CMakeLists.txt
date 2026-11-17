# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Declares smoke.pte and its kernel bundle to the executorch component, and
# embeds the model with its test cases (smoke.io) in the table main.c walks.
idf_component_register(
  SRCS "main.c" "${CMAKE_CURRENT_BINARY_DIR}/models.S" PRIV_REQUIRES executorch
)
executorch_add_model(PTE "${CMAKE_CURRENT_SOURCE_DIR}/smoke.pte")
executorch_add_kernels(DIR "${CMAKE_CURRENT_SOURCE_DIR}/smoke_kernels")

file(
  CONFIGURE
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/models.S"
  CONTENT
  "    .section .rodata.smoke_models, \"a\"
    .balign 4
    .global smoke_models
smoke_models:
    .word name, pte, pte_end, io
    .global smoke_model_count
smoke_model_count:
    .word 1
name:
    .asciz \"smoke\"
    .balign 16
pte:
    .incbin \"${CMAKE_CURRENT_SOURCE_DIR}/smoke.pte\"
pte_end:
    .balign 16
io:
    .incbin \"${CMAKE_CURRENT_SOURCE_DIR}/smoke.io\"
    .section .note.GNU-stack, \"\", @progbits
"
)
set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/models.S"
  PROPERTIES
    OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/smoke.pte;${CMAKE_CURRENT_SOURCE_DIR}/smoke.io"
)
