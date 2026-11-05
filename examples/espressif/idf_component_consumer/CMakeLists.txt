# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The idf_component_smoke consumer, but with executorch resolved by the
# component manager, the way a project using the published component gets it.
# Pass the version to resolve as ET_COMPONENT_VERSION and the smoke project's
# ET_SMOKE_* variables. Used by .ci/scripts/test_espressif_idf_consumer.sh.

cmake_minimum_required(VERSION 3.16)

if(NOT ET_COMPONENT_VERSION)
  message(FATAL_ERROR "Set ET_COMPONENT_VERSION")
endif()
# The component manager reads main/idf_component.yml before any component's
# CMakeLists.txt runs, and does not expand variables in versions.
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_LIST_DIR}/main/idf_component.yml"
     CONTENT
     "dependencies:\n  pytorch/executorch: \"==${ET_COMPONENT_VERSION}\"\n"
)
set(SDKCONFIG_DEFAULTS
    "${CMAKE_CURRENT_LIST_DIR}/../idf_component_smoke/sdkconfig.defaults"
)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(et_component_smoke)
