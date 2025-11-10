# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# lint_cmake: -linelength
add_library(
  cadence_kernels
  kernels.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/library/api/tensor_transposef.c
  ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/library/api/vsoftmaxf.c
  ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/library/tables/expf_tbl.c
  ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/library/tables/nanf_tbl.c
  ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/library/tables/inff_tbl.c
)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_include_directories(
  cadence_kernels
  PUBLIC . ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/include
         ${EXECUTORCH_ROOT}/backends/cadence/vision/third-party/include_private
         ${_common_include_directories}
)

target_link_libraries(cadence_kernels PRIVATE idma)
