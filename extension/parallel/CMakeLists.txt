# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please keep this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

if(NOT (EXECUTORCH_BUILD_PTHREADPOOL AND EXECUTORCH_BUILD_CPUINFO))
  message(FATAL_ERROR "extension/parallel requires extension/threadpool")
endif()

add_library(extension_parallel thread_parallel.cpp)

target_link_libraries(extension_parallel PUBLIC executorch_core extension_threadpool)
target_compile_options(extension_parallel PUBLIC ${_common_compile_options})

install(
  TARGETS extension_parallel
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories})
