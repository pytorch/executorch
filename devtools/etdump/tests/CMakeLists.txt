# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs etdump_test.cpp)

# etdump_test.cpp includes etdump_filter.h, which includes <re2/re2.h>. re2 only
# reaches CMake through the tokenizers submodule, which a default checkout does
# not populate, so re2::re2 usually does not exist and this suite is skipped
# here. Buck builds it unconditionally, which is where it runs today.
#
# The guard is not sufficient on its own: this target links neither re2 nor
# etdump_filter.cpp (the etdump library does not compile that file), so where
# re2::re2 does exist the suite fails to link. That predates this file's change,
# which only narrows when it is attempted. Fixing it properly means adding
# etdump_filter.cpp to the etdump library and re2 to EXTRA_LIBS.
#
# The two tests below avoid etdump_filter.h so they always build.
if(TARGET re2::re2)
  et_cxx_test(
    sdk_etdump_tests
    SOURCES
    ${_test_srcs}
    EXTRA_LIBS
    bundled_program
    etdump
    flatccrt
  )
  target_include_directories(
    sdk_etdump_tests PRIVATE ${CMAKE_INSTALL_PREFIX}/sdk/include
                             ${EXECUTORCH_ROOT}/third-party/flatcc/include
  )
endif()

et_cxx_test(
  etdump_device_test SOURCES etdump_device_test.cpp EXTRA_LIBS etdump flatccrt
)

et_cxx_test(
  etdump_device_no_allocator_test SOURCES etdump_device_no_allocator_test.cpp
  EXTRA_LIBS etdump flatccrt
)

# These two register a device allocator and then call into etdump, which reads
# it back out of the registry. et_cxx_test links the static executorch_core, so
# in a shared build, where etdump resolves the runtime from libexecutorch.so,
# the archive would win the runtime symbols and the test would hold a private
# registry (Utils.cmake documents that ordering). The usual remedy,
# executorch_target_link_shared_runtime, cannot be called from here: the root
# CMakeLists adds devtools well before it defines executorch_shared, so the
# target does not exist yet at this point. These tests are run by the Buck
# targets and by the default static CMake build, both of which have one
# registry; a shared-build CMake run of them needs that ordering addressed
# first.
