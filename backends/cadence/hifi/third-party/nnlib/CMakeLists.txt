cmake_minimum_required(VERSION 3.10.0)
project(cadence_nnlib)

if(EXECUTORCH_HIFI_CORE STREQUAL "hifi5")
  set(NNLIB_DIR_NAME "nnlib-hifi5")
else()
  set(NNLIB_DIR_NAME "nnlib-hifi4")
endif()

add_custom_target(
  nnlib_target ALL
  COMMAND
    make install_nnlib -f makefile -C
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/${NNLIB_DIR_NAME}/xa_nnlib/build
    OBJDIR=${CMAKE_CURRENT_BINARY_DIR}/obj
    LIBDIR=${CMAKE_CURRENT_BINARY_DIR}/lib -j8
)

add_library(xa_nnlib STATIC IMPORTED GLOBAL)
add_dependencies(xa_nnlib nnlib_target)

set_property(
  TARGET xa_nnlib PROPERTY IMPORTED_LOCATION
                           "${CMAKE_CURRENT_BINARY_DIR}/lib/xa_nnlib.a"
)
