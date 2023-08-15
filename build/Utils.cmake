# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is intended to have helper functions to keep the CMakeLists.txt concise. If there are any helper function can be re-used, it's recommented to add them here.


# Public function to print summary for all configurations. For new variable, it's recommended to add them here.
function(executorch_print_configuration_summary)
    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "  BUCK                          : ${BUCK2}")
    message(STATUS "  CMAKE_CXX_STANDARD            : ${CMAKE_CXX_STANDARD}")
    message(STATUS "  CMAKE_CXX_COMPILER_ID         : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  CMAKE_TOOLCHAIN_FILE          : ${CMAKE_TOOLCHAIN_FILE}")
    message(STATUS "  FLATBUFFERS_BUILD_FLATC       : ${FLATBUFFERS_BUILD_FLATC}")
    message(STATUS "  FLATBUFFERS_BUILD_FLATHASH    : ${FLATBUFFERS_BUILD_FLATHASH}")
    message(STATUS "  FLATBUFFERS_BUILD_FLATLIB     : ${FLATBUFFERS_BUILD_FLATLIB}")
    message(STATUS "  FLATBUFFERS_BUILD_TESTS       : ${FLATBUFFERS_BUILD_TESTS}")
    message(STATUS "  REGISTER_EXAMPLE_CUSTOM_OPS   : ${REGISTER_EXAMPLE_CUSTOM_OPS}")
endfunction()

# This is the funtion to use -Wl to link static library, used for clang
function(clang_kernel_link_options target_name)
    target_link_options(${target_name}
        INTERFACE
        # TODO(dbort): This will cause the .a to show up on the link line twice
        # for targets that depend on this library; once because CMake will add
        # it, and once because it's in this list of args. See if there's a way
        # to avoid that.
        -Wl,-force_load,$<TARGET_FILE:${target_name}>
    )
endfunction()

# This is the funtion to use -Wl, --whole-archive to link static library, used for gcc
function(gcc_kernel_link_options target_name)
    target_link_options(${target_name}
        INTERFACE
        # TODO(dbort): This will cause the .a to show up on the link line twice
        -Wl,--whole-archive
        $<TARGET_FILE:${target_name}>
        -Wl,--no-whole-archive
    )
endfunction()
