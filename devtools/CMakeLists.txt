# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The include directory that will contain the generated schema headers.
set(DEVTOOLS_INCLUDE_DIR_NO_BUILD_INTERFACE
    ${CMAKE_BINARY_DIR}/devtools/include
)
set(DEVTOOLS_INCLUDE_DIR
    $<BUILD_INTERFACE:${DEVTOOLS_INCLUDE_DIR_NO_BUILD_INTERFACE}>
)

add_subdirectory(etdump)
add_subdirectory(bundled_program)
