# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# With ET_CONSUMER_MODELS_DIR, runs the models CONFIG_ET_CONSUMER_MODEL lists
# from it, with their kernel bundle.
if(ET_CONSUMER_MODELS_DIR)
  set(ET_SMOKE_MODELS "")
  foreach(model IN LISTS CONFIG_ET_CONSUMER_MODEL)
    list(APPEND ET_SMOKE_MODELS "${ET_CONSUMER_MODELS_DIR}/${model}.pte")
  endforeach()
  string(REPLACE ";" "_" bundle "${CONFIG_ET_CONSUMER_MODEL}")
  set(ET_SMOKE_KERNELS "${ET_CONSUMER_MODELS_DIR}/${bundle}_kernels")
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../../idf_component_smoke/main/CMakeLists.txt)
