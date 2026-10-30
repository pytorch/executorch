# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)

get_filename_component(
  EXECUTORCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
set(EXTRA_COMPONENT_DIRS "${EXECUTORCH_ROOT}/backends/espressif/espnn_runtime")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(camera_stream)
