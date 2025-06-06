# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FetchContent gflags and make available at CMake configure time.
# Make the content available at configure time so that BUCK can use it.
include(FetchContent)

function(FetchContent_gflags)
  FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
  )
  set(GFLAGS_INTTYPES_FORMAT C99)
  FetchContent_MakeAvailable(gflags)
endfunction()
