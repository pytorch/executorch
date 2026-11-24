# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FetchContent)

set(ESPNN_GIT_TAG
    "v1.5.0"
    CACHE STRING "ESP-NN revision to fetch when no local checkout is given"
)
set(ESPNN_LOCAL_PATH
    ""
    CACHE PATH "Existing ESP-NN checkout to build instead of fetching one"
)

if(ESPNN_LOCAL_PATH)
  set(espnn_source_dir "${ESPNN_LOCAL_PATH}")
else()
  # Fetch sources without adding ESP-NN's ESP-IDF component CMake project.
  FetchContent_Declare(
    esp_nn
    GIT_REPOSITORY https://github.com/espressif/esp-nn.git
    GIT_TAG ${ESPNN_GIT_TAG}
    SOURCE_SUBDIR none
  )
  FetchContent_MakeAvailable(esp_nn)
  set(espnn_source_dir "${esp_nn_SOURCE_DIR}")
endif()
