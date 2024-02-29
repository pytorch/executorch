# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
enable_testing()
find_package(Torch CONFIG REQUIRED)
# find_package(GTest REQUIRED)
add_executable(
  make_aten_functor_from_et_functor_test
  make_aten_functor_from_et_functor_test.cpp
)
target_link_libraries(
  make_aten_functor_from_et_functor_test
  executorch
  torch
  GTest::gtest_main
)
target_include_directories(make_aten_functor_from_et_functor_test PUBLIC GTEST_INCLUDE_DIRS TORCH_INCLUDE_DIRS)
target_compile_options(make_aten_functor_from_et_functor_test PUBLIC -frtti -fexceptions)

include(GoogleTest)
gtest_discover_tests(make_aten_functor_from_et_functor_test)
