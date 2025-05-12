# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(_is_build_type_release ON)
  set(_is_build_type_debug OFF)
else()
  set(_is_build_type_release OFF)
  set(_is_build_type_debug ON)
endif()

# MARK: - Overridable Options

define_overridable_option(EXECUTORCH_ENABLE_LOGGING "Build with ET_LOG_ENABLED" BOOL ${_is_build_type_debug})
define_overridable_option(EXECUTORCH_BUILD_COREML "Build the Core ML backend" BOOL OFF)
define_overridable_option(
  EXECUTORCH_FLATBUFFERS_MAX_ALIGNMENT
  "Exir lets users set the alignment of tensor data embedded in the flatbuffer, and some users need an alignment larger than the default, which is typically 32."
  STRING 1024
)
