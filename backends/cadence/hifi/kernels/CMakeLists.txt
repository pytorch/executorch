# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# lint_cmake: -linelength
add_library(
  cadence_kernels
  kernels.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/matmul_asym8uxasym8u_asym8u.cpp
)

target_include_directories(
  cadence_kernels
  PUBLIC
    .
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/algo/common/include/
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/include/nnlib
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/include
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/algo/ndsp/hifi4/include/
)

target_link_libraries(cadence_kernels PRIVATE xa_nnlib)
