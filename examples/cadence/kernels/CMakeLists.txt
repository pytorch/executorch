# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# lint_cmake: -linelength
add_library(cadence_kernels kernels.cpp ${EXECUTORCH_ROOT}/examples/cadence/third-party/nnlib-hifi4/matmul_asym8uxasym8u_asym8u.cpp)

target_include_directories(
  cadence_kernels
  PUBLIC .
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/common/include/
         ${NN_LIB_BASE_DIR}/xa_nnlib/include/nnlib
         ${NN_LIB_BASE_DIR}/xa_nnlib/include
         ${NN_LIB_BASE_DIR}/xa_nnlib/algo/ndsp/hifi4/include/
         ${NXP_SDK_ROOT_DIR}/middleware/dsp/naturedsp/hifi4/include/)
