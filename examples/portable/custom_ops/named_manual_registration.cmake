# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GNUInstallDirs)

function(add_manual_registration_lib lib_name op_name kernel_source)
  gen_selected_ops(LIB_NAME "${lib_name}" ROOT_OPS "${op_name}")
  generate_bindings_for_kernels(
    LIB_NAME "${lib_name}" CUSTOM_OPS_YAML
    ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/custom_ops.yaml MANUAL_REGISTRATION
  )
  add_library(${lib_name}_kernels ${kernel_source})
  target_link_libraries(${lib_name}_kernels PRIVATE executorch)
  gen_operators_lib(
    LIB_NAME
    "${lib_name}"
    KERNEL_LIBS
    ${lib_name}_kernels
    DEPS
    executorch
    MANUAL_REGISTRATION
  )
  install(
    TARGETS ${lib_name} ${lib_name}_kernels
    EXPORT ManualRegistrationTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch
  )
endfunction()

add_manual_registration_lib(
  manual_ops_1_lib "my_ops::mul3.out"
  ${CMAKE_CURRENT_LIST_DIR}/custom_ops_1_out.cpp
)
add_manual_registration_lib(
  manual_ops_2_lib "my_ops::mul4.out"
  ${CMAKE_CURRENT_LIST_DIR}/custom_ops_2_out.cpp
)

# Use the same kernel implementation so the failure is a registry collision, not
# a duplicate C++ definition or a link failure.
gen_selected_ops(LIB_NAME manual_ops_overlap_lib ROOT_OPS "my_ops::mul3.out")
generate_bindings_for_kernels(
  LIB_NAME manual_ops_overlap_lib CUSTOM_OPS_YAML
  ${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml MANUAL_REGISTRATION
)
gen_operators_lib(
  LIB_NAME
  manual_ops_overlap_lib
  KERNEL_LIBS
  manual_ops_1_lib_kernels
  DEPS
  executorch
  MANUAL_REGISTRATION
)

add_executable(
  named_manual_registration_test
  ${CMAKE_CURRENT_LIST_DIR}/named_manual_registration_test.cpp
)
target_link_libraries(
  named_manual_registration_test PRIVATE manual_ops_1_lib manual_ops_2_lib
                                         manual_ops_overlap_lib
)

install(EXPORT ManualRegistrationTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ManualRegistration
)
