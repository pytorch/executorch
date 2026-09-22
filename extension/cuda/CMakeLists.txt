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
add_library(extension_cuda SHARED caller_stream.cpp device_guard.cpp)
if(EXECUTORCH_BUILD_SHARED)
  # Named after what it provides rather than after the target, matching the
  # other libraries the wheel ships. The target name stays as it is because the
  # rest of the build already refers to it.
  set_target_properties(
    extension_cuda PROPERTIES OUTPUT_NAME executorch_extension_cuda
  )
  executorch_target_soname_policy(extension_cuda)
  # Ships beside libexecutorch.so in the wheel's lib/ directory.
  executorch_target_shipped_runtime_path(extension_cuda)
endif()
# device_guard.cpp calls into the runtime, so this needs it. Only built under an
# accelerator build, where the toolkit is present by definition.
# caller_stream.cpp still calls nothing and treats its handle as opaque.
if(EXECUTORCH_BUILD_ROCM)
  target_link_libraries(extension_cuda PUBLIC hip::host)
else()
  target_link_libraries(extension_cuda PUBLIC CUDA::cudart)
endif()
# device_guard.cpp logs, so this needs the ExecuTorch runtime. Taken through the
# helper rather than as an ordinary dependency, because CMake would order the
# static core ahead of the shared runtime and leave this library with its own
# copy of the registries.
executorch_target_link_shared_runtime(extension_cuda)
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
