# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The idf_component_smoke consumer, but with executorch resolved by the
# component manager, the way a project using the published component gets it.
# Pass the version to resolve as ET_COMPONENT_VERSION, and either the smoke
# project's ET_SMOKE_* variables, as .ci/scripts/test_espressif_idf_consumer.sh
# does, or ET_CONSUMER_MODELS_DIR, from which CONFIG_ET_CONSUMER_MODEL (set by
# the sdkconfig.ci.* configurations) picks the models, as
# .ci/scripts/test_espressif_idf_nightly.sh does. ET_COMPONENT_REGISTRY_URL
# picks the registry of pytorch/executorch, instead of the component manager's
# default; esp-nn comes from the registry the component's manifest names.

cmake_minimum_required(VERSION 3.16)

if(NOT ET_COMPONENT_VERSION)
  message(FATAL_ERROR "Set ET_COMPONENT_VERSION")
endif()
set(ET_REGISTRY_LINE "")
if(ET_COMPONENT_REGISTRY_URL)
  set(ET_REGISTRY_LINE "\n    registry_url: ${ET_COMPONENT_REGISTRY_URL}")
endif()
# The component manager reads main/idf_component.yml before any component's
# CMakeLists.txt runs, and does not expand variables in versions.
file(
  CONFIGURE
  OUTPUT
  "${CMAKE_CURRENT_LIST_DIR}/main/idf_component.yml"
  CONTENT
  "dependencies:\n  pytorch/executorch:\n    version: \"==${ET_COMPONENT_VERSION}\"${ET_REGISTRY_LINE}\n"
)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(et_component_smoke)
