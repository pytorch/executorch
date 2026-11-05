# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(ET_SMOKE_PTE)
  set(srcs "main.c" "model.S")
else()
  set(srcs "empty.c")
endif()
idf_component_register(SRCS ${srcs} PRIV_REQUIRES executorch)
if(NOT ET_SMOKE_PTE)
  return()
endif()

set_source_files_properties(
  model.S PROPERTIES COMPILE_DEFINITIONS "ET_SMOKE_PTE=\"${ET_SMOKE_PTE}\""
)
if(NOT ET_SMOKE_UNDECLARED)
  executorch_add_model(PTE "${ET_SMOKE_PTE}")
endif()
if(ET_SMOKE_KERNELS)
  executorch_add_kernels(DIR "${ET_SMOKE_KERNELS}")
endif()
