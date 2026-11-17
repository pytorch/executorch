# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Paths are built from this file's directory, since idf_component_consumer
# includes it.
if(ET_SMOKE_MODELS)
  set(srcs "${CMAKE_CURRENT_LIST_DIR}/main.c"
           "${CMAKE_CURRENT_BINARY_DIR}/models.S"
  )
else()
  set(srcs "${CMAKE_CURRENT_LIST_DIR}/empty.c")
endif()
idf_component_register(SRCS ${srcs} PRIV_REQUIRES executorch)
if(NOT ET_SMOKE_MODELS)
  return()
endif()

# Embeds each model and the .io file split_bundled_program.py wrote next to it,
# with the table main.c walks.
set(table "")
set(data "")
set(files "")
set(index 0)
foreach(pte IN LISTS ET_SMOKE_MODELS)
  get_filename_component(pte "${pte}" ABSOLUTE)
  get_filename_component(name "${pte}" NAME_WLE)
  get_filename_component(dir "${pte}" DIRECTORY)
  set(io "${dir}/${name}.io")
  string(
    APPEND table
    "    .word name_${index}, pte_${index}, pte_end_${index}, io_${index}\n"
  )
  string(
    APPEND data "name_${index}:\n    .asciz \"${name}\"\n"
    "    .balign 16\npte_${index}:\n    .incbin \"${pte}\"\npte_end_${index}:\n"
    "    .balign 16\nio_${index}:\n    .incbin \"${io}\"\n"
  )
  list(APPEND files "${pte}" "${io}")
  if(NOT ET_SMOKE_BACKENDS AND NOT ET_SMOKE_UNDECLARED)
    executorch_add_model(PTE "${pte}")
  endif()
  math(EXPR index "${index} + 1")
endforeach()
file(
  CONFIGURE
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/models.S"
  CONTENT
  "    .section .rodata.smoke_models, \"a\"\n    .balign 4\n    .global smoke_models\nsmoke_models:\n${table}    .global smoke_model_count\nsmoke_model_count:\n    .word ${index}\n${data}\n    .section .note.GNU-stack, \"\", @progbits\n"
)
set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/models.S" PROPERTIES OBJECT_DEPENDS "${files}"
)
if(ET_SMOKE_KERNELS)
  executorch_add_kernels(DIR "${ET_SMOKE_KERNELS}")
endif()
foreach(backend IN LISTS ET_SMOKE_BACKENDS)
  executorch_add_backend(${backend})
endforeach()
