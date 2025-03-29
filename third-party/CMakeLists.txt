# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# MARK: - Flatbuffers

set(FLATBUFFERS_BUILD_FLATC ON)
set(FLATBUFFERS_BUILD_FLATHASH OFF)
set(FLATBUFFERS_BUILD_FLATLIB ON)
set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)

add_subdirectory(flatbuffers)
