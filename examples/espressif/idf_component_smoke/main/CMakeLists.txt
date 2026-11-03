# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

idf_component_register(SRCS "main.c" "model.S" PRIV_REQUIRES executorch)
set_source_files_properties(
  model.S PROPERTIES COMPILE_DEFINITIONS "ET_SMOKE_PTE=\"${ET_SMOKE_PTE}\""
)
