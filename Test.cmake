# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# A helper CMake file to trigger C++ unit tests.
#

if(BUILD_TESTING)
  # This contains the list of tests which are always built
  add_subdirectory(extension/evalue_util/test)
  add_subdirectory(extension/kernel_util/test)
  add_subdirectory(extension/memory_allocator/test)
  add_subdirectory(extension/parallel/test)
  add_subdirectory(extension/pytree/test)
  add_subdirectory(kernels/portable/cpu/util/test)
  add_subdirectory(kernels/prim_ops/test)
  add_subdirectory(kernels/test)
  add_subdirectory(runtime/core/exec_aten/testing_util/test)
  add_subdirectory(runtime/core/exec_aten/util/test)
  add_subdirectory(runtime/core/portable_type/test)
  add_subdirectory(runtime/core/test)
  add_subdirectory(runtime/executor/test)
  add_subdirectory(runtime/kernel/test)
  add_subdirectory(runtime/platform/test)
  add_subdirectory(test/utils)
endif()
