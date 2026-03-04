# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example ESP-IDF project CMakeLists.txt
#
# This is a template project that uses the executor_runner component.
# Copy this to your own project directory and adjust paths as needed.
#
# Usage:
#   cd examples/espressif/project
#   idf.py set-target esp32s3
#   idf.py build
#   idf.py flash monitor

cmake_minimum_required(VERSION 3.16)

# Set the path to ExecuTorch source
set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "ExecuTorch root")

# Add the executor_runner as an extra component
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../executor_runner"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(executorch_esp_runner)
