# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The logic is copied from
# https://github.com/pytorch/pytorch/blob/main/cmake/Dependencies.cmake
set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")

# --- XNNPACK

set(XNNPACK_SOURCE_DIR "${THIRD_PARTY_ROOT}/XNNPACK")
set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include")

include(ExternalProject)
include(GNUInstallDirs) # For CMAKE_INSTALL_LIBDIR
set(XNNPACK_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/XNNPACK/install")
set(XNNPACK_STATIC_LIB
    "${XNNPACK_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libXNNPACK.a"
)
set(XNNPACK_MICROKERNELS_STATIC_LIB
    "${XNNPACK_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libxnnpack-microkernels-prod.a"
)
set(KLEIDIAI_STATIC_LIB
    "${XNNPACK_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libkleidiai.a"
)
get_extra_cmake_args_for_external_project(XNNPACK_EXTRA_CMAKE_ARGS)
ExternalProject_Add(
  XNNPACKExternalProject
  SOURCE_DIR ${XNNPACK_SOURCE_DIR}
  # Not 100% clear on these locations
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/XNNPACK
  INSTALL_DIR ${XNNPACK_INSTALL_DIR}
  INSTALL_BYPRODUCTS ${XNNPACK_STATIC_LIB} ${XNNPACK_MICROKERNELS_STATIC_LIB}
  ${KLEIDIAI_STATIC_LIB}
  CMAKE_ARGS
    ${XNNPACK_EXTRA_CMAKE_ARGS}
    -D
    XNNPACK_LIBRARY_TYPE=static
    -D
    XNNPACK_BUILD_BENCHMARKS=OFF
    -D
    XNNPACK_BUILD_TESTS=OFF
    -D
    XNNPACK_ENABLE_AVXVNNI=OFF
    # Work around observed failure:
    # https://github.com/pytorch/executorch/pull/10362#issuecomment-2906391232
    -D
    XNNPACK_ENABLE_AVX512VNNIGFNI=OFF
    -D
    ENABLE_XNNPACK_WEIGHTS_CACHE=${EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE}
    -D
    ENABLE_XNNPACK_SHARED_WORKSPACE=${EXECUTORCH_XNNPACK_SHARED_WORKSPACE}
    -D
    XNNPACK_ENABLE_KLEIDIAI=${EXECUTORCH_XNNPACK_ENABLE_KLEIDI}
    -D
    CMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -D
    XNNPACK_BUILD_ALL_MICROKERNELS=OFF
    -D
    CMAKE_POSITION_INDEPENDENT_CODE=ON
)

add_library(XNNPACK STATIC IMPORTED GLOBAL)
# TODO: this probably doesn't work on Windows.
set_property(TARGET XNNPACK PROPERTY IMPORTED_LOCATION ${XNNPACK_STATIC_LIB})

add_dependencies(XNNPACK XNNPACKExternalProject)

add_library(xnnpack-microkernels-prod STATIC IMPORTED GLOBAL)
set_property(
  TARGET xnnpack-microkernels-prod PROPERTY IMPORTED_LOCATION
                                            ${XNNPACK_MICROKERNELS_STATIC_LIB}
)
add_dependencies(xnnpack-microkernels-prod XNNPACKExternalProject)

set_target_properties(
  XNNPACK PROPERTIES INTERFACE_LINK_LIBRARIES xnnpack-microkernels-prod
)

install(DIRECTORY ${XNNPACK_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX})

if(EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  add_library(kleidiai SHARED IMPORTED)
  install(FILES ${KLEIDIAI_STATIC_LIB} PUBLIC_HEADER
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  set_property(
    TARGET kleidiai PROPERTY IMPORTED_LOCATION ${KLEIDIAI_STATIC_LIB}
  )
  set_property(
    TARGET XNNPACK
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES kleidiai
  )
endif()
