# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please keep this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

find_package(CUDAToolkit REQUIRED)

# SHARED on purpose: the caller-stream thread-local must have a single
# definition across every shared object in the process (see export.h). A static
# copy linked into multiple shared libraries would create multiple thread-locals
# and silently break the caller-stream handshake.
add_library(extension_cuda SHARED caller_stream.cpp)
target_link_libraries(extension_cuda PUBLIC CUDA::cudart)
target_include_directories(extension_cuda PUBLIC ${_common_include_directories})
target_compile_options(
  extension_cuda PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_common_compile_options}>"
)
target_compile_definitions(
  extension_cuda PRIVATE EXECUTORCH_EXTENSION_CUDA_BUILDING
)

# A namespaced, versioned name, matching the other libraries the wheel ships.
# The CUDA delegate carries a real symbol reference to this library, so its
# DT_NEEDED entry survives even under --as-needed. A generic name like
# libextension_cuda.so could be satisfied by an unrelated library that happens
# to be loaded first, which would bind a different caller-stream implementation
# into the delegate.
#
# Not gated on the shared runtime: the target above is always SHARED, so gating
# the name would produce a different file name in a static-runtime build than
# packaging looks for.
set_target_properties(
  extension_cuda
  PROPERTIES OUTPUT_NAME executorch_extension_cuda
             VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
)

install(
  TARGETS extension_cuda
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
