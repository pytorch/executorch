# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(native_runner_utils STATIC runner_utils.cpp)
target_compile_features(native_runner_utils PUBLIC cxx_std_20)
target_compile_options(
  native_runner_utils PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                              $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
)
target_include_directories(
  native_runner_utils PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
)
