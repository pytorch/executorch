# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimal ESP-IDF consumer of the generated ExecuTorch component, used by
# .ci/scripts/test_espressif_idf_component.sh. Pass the generated component
# directory as ET_COMPONENT_DIR and the .pte files that split_bundled_program.py
# made as the ET_SMOKE_MODELS list; each runs on its bundled test cases. Pass a
# kernel bundle from the component's tools/gen_kernels.py as ET_SMOKE_KERNELS,
# or leave it out to have the build make one. With ET_SMOKE_BACKENDS, the models
# are embedded but not declared and only those delegates are, as for models
# delivered over the air; with ET_SMOKE_UNDECLARED they are embedded but not
# declared. Without ET_SMOKE_MODELS it builds an app that does not use
# ExecuTorch, for size comparisons.

cmake_minimum_required(VERSION 3.16)

if(NOT ET_COMPONENT_DIR)
  message(FATAL_ERROR "Set ET_COMPONENT_DIR")
endif()
set(EXTRA_COMPONENT_DIRS "${ET_COMPONENT_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(et_component_smoke)
