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

add_executable(
  make_boxed_from_unboxed_functor_test
  make_boxed_from_unboxed_functor_test.cpp
)
target_link_libraries(
  make_boxed_from_unboxed_functor_test
  executorch
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(make_boxed_from_unboxed_functor_test)
