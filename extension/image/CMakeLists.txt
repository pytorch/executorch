# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

if(APPLE)
  enable_language(OBJCXX)
  add_library(
    extension_image image_processor_common.cpp image_processor_simd.cpp
                    image_processor_apple.cpp image_processor_apple_gpu.mm
  )
  set_source_files_properties(
    image_processor_apple_gpu.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc"
  )
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
  find_library(COREIMAGE_FRAMEWORK CoreImage REQUIRED)
  find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(
    extension_image
    PRIVATE ${ACCELERATE_FRAMEWORK} ${COREGRAPHICS_FRAMEWORK}
            ${COREIMAGE_FRAMEWORK} ${COREVIDEO_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
  )
else()
  # stb_image_resize: lightweight header-only library used by the resize step in
  # image_processor.cpp. Only the portable (non-Apple) path uses stb; the Apple
  # path resizes via vImage, so the fetch is scoped here to avoid downloading
  # stb on Apple builds.
  include(FetchContent)
  FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
  )
  FetchContent_MakeAvailable(stb)

  add_library(
    extension_image image_processor_common.cpp image_processor_simd.cpp
                    image_processor.cpp
  )

  # stb_image_resize.h lives under deprecated/ in current stb. Private: only the
  # .cpp uses it, not the installed public headers.
  target_include_directories(
    extension_image PRIVATE ${stb_SOURCE_DIR} ${stb_SOURCE_DIR}/deprecated
  )
endif()

target_include_directories(
  extension_image PUBLIC ${_common_include_directories}
)

target_link_libraries(extension_image PUBLIC executorch_core extension_tensor)

install(
  TARGETS extension_image
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(APPLE)
  install(FILES image_processor.h image_processor_config.h
                image_processor_apple.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/image
  )
else()
  install(FILES image_processor.h image_processor_config.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/extension/image
  )
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
