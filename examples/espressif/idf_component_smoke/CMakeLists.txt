# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimal ESP-IDF consumer of the generated ExecuTorch component, used by
# .ci/scripts/test_espressif_idf_component.sh. Pass the generated component
# directory as ET_COMPONENT_DIR, the plain smoke .pte from export_smoke_model.py
# as ET_SMOKE_PTE, and its kernel bundle from the component's
# tools/gen_kernels.py as ET_SMOKE_KERNELS. ET_SMOKE_UNDECLARED runs the model
# without declaring it. Without ET_SMOKE_PTE it builds an app that does not use
# ExecuTorch, for size comparisons.

cmake_minimum_required(VERSION 3.16)

if(NOT ET_COMPONENT_DIR)
  message(FATAL_ERROR "Set ET_COMPONENT_DIR")
endif()
set(EXTRA_COMPONENT_DIRS "${ET_COMPONENT_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(et_component_smoke)
