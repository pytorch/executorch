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

if(EXECUTORCH_BUILD_ROCM)
  find_package(hip CONFIG REQUIRED)
endif()

# SHARED on purpose: the caller-stream thread-local must have a single
# definition across every shared object in the process (see export.h). A static
# copy linked into multiple shared libraries would create multiple thread-locals
# and silently break the caller-stream handshake.
add_library(extension_cuda SHARED caller_stream.cpp)
if(EXECUTORCH_BUILD_SHARED)
  # Named after what it provides rather than after the target, matching the
  # other libraries the wheel ships. The target name stays as it is because the
  # rest of the build already refers to it.
  set_target_properties(
    extension_cuda PROPERTIES OUTPUT_NAME executorch_extension_cuda
  )
  executorch_target_soname_policy(extension_cuda)
endif()
# No CUDA headers or libraries: caller_stream.cpp uses cudaStream_t as an opaque
# handle and calls no CUDA function, so it compiles without cuda_runtime.h and
# needs no libcudart. Anything else that links this library and does call CUDA
# links the runtime itself. ROCm is the exception: the handle is a hip type
# there, so the header needs hip's own.
if(EXECUTORCH_BUILD_ROCM)
  target_link_libraries(extension_cuda PUBLIC hip::host)
endif()
target_include_directories(extension_cuda PUBLIC ${_common_include_directories})
target_compile_options(
  extension_cuda PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${_common_compile_options}>"
)
target_compile_definitions(
  extension_cuda PRIVATE EXECUTORCH_EXTENSION_CUDA_BUILDING
)
if(EXECUTORCH_BUILD_ROCM)
  target_compile_definitions(extension_cuda PUBLIC EXECUTORCH_USE_HIP=1)
endif()

install(
  TARGETS extension_cuda
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${_common_include_directories}
)
