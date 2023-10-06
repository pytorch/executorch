# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(bundled_program
            ${CMAKE_CURRENT_SOURCE_DIR}/bundled_program_verification.cpp)

target_link_libraries(bundled_program executorch bundled_schema)
