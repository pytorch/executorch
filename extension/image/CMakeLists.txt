# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

# stb_image_resize: lightweight header-only library used by the resize step in
# image_processor.cpp.
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)

add_library(extension_image image_processor_common.cpp image_processor.cpp)

target_include_directories(
  extension_image PUBLIC ${_common_include_directories}
)

# stb_image_resize.h lives under deprecated/ in current stb. Private: only the
# .cpp uses it, not the installed public headers.
target_include_directories(
  extension_image PRIVATE ${stb_SOURCE_DIR} ${stb_SOURCE_DIR}/deprecated
)

target_link_libraries(extension_image PUBLIC executorch_core extension_tensor)

install(
  TARGETS extension_image
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES image_processor.h image_processor_config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/image
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
