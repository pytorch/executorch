# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# With ET_CONSUMER_MODELS_DIR, runs CONFIG_ET_CONSUMER_MODEL's .pte and kernel
# bundle from it.
if(ET_CONSUMER_MODELS_DIR)
  set(ET_SMOKE_MODELS
      "${ET_CONSUMER_MODELS_DIR}/${CONFIG_ET_CONSUMER_MODEL}.pte"
  )
  set(ET_SMOKE_KERNELS
      "${ET_CONSUMER_MODELS_DIR}/${CONFIG_ET_CONSUMER_MODEL}_kernels"
  )
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../../idf_component_smoke/main/CMakeLists.txt)
