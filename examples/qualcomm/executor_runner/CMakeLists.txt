# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_qnn_executor_runner__srcs ${_executor_runner__srcs})

# preprocess executor runner src files
list(TRANSFORM _qnn_executor_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _qnn_executor_runner__srcs EXCLUDE REGEX ".*executor_runner.cpp$")
list(PREPEND _qnn_executor_runner__srcs
     ${CMAKE_CURRENT_LIST_DIR}/qnn_executor_runner.cpp
)

# build executor runner
add_executable(qnn_executor_runner ${_qnn_executor_runner__srcs})
target_include_directories(
  qnn_executor_runner
  PUBLIC ${_common_include_directories}
         # For <flatcc/...>, used directly by the runner to free etdump's
         # aligned buffer. It also arrives transitively through the etdump
         # target in an in-tree build, but name it here so the dependency is
         # explicit and does not rely on that export.
         ${EXECUTORCH_ROOT}/third-party/flatcc/include
)
target_link_libraries(
  qnn_executor_runner
  qnn_executorch_backend
  executorch_core
  extension_evalue_util
  extension_runner_util
  full_portable_ops_lib
  quantized_ops_lib
  etdump
  flatccrt
  gflags
)
set_target_properties(
  qnn_executor_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
