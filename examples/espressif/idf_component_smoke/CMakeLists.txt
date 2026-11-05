# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimal ESP-IDF consumer of the generated ExecuTorch component, used by
# .ci/scripts/test_espressif_idf_component.sh. Pass the generated component
# directory as ET_COMPONENT_DIR and the plain smoke .pte from
# export_smoke_model.py as ET_SMOKE_PTE.

cmake_minimum_required(VERSION 3.16)

if(NOT ET_COMPONENT_DIR OR NOT ET_SMOKE_PTE)
  message(FATAL_ERROR "Set ET_COMPONENT_DIR and ET_SMOKE_PTE")
endif()
set(EXTRA_COMPONENT_DIRS "${ET_COMPONENT_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(et_component_smoke)
