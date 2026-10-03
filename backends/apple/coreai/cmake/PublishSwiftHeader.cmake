# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if("$ENV{DERIVED_SOURCES_DIR}" STREQUAL "" OR "$ENV{CONFIGURATION}" STREQUAL "")
  message(FATAL_ERROR "Swift header publication requires Xcode build settings")
endif()
set(_header "$ENV{DERIVED_SOURCES_DIR}/$ENV{SWIFT_OBJC_INTERFACE_HEADER_NAME}")
set(_destination
    "${PUBLISH_ROOT}/$ENV{CONFIGURATION}$ENV{EFFECTIVE_PLATFORM_NAME}"
)
file(MAKE_DIRECTORY "${_destination}")
file(COPY_FILE "${_header}" "${_destination}/CoreAIBridge-Swift.h"
     ONLY_IF_DIFFERENT
)
