# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#
# Build tokenizers.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name} INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive")
endfunction()

# Same as kernel_link_options but it's for MacOS linker
function(macos_kernel_link_options target_name)
  target_link_options(${target_name} INTERFACE
                      "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>")
endfunction()

# Same as kernel_link_options but it's for MSVC linker
function(msvc_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:/WHOLEARCHIVE:$<TARGET_FILE:${target_name}>")
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(target_link_options_shared_lib target_name)
  if(APPLE)
    macos_kernel_link_options(${target_name})
  elseif(MSVC)
    msvc_kernel_link_options(${target_name})
  else()
    kernel_link_options(${target_name})
  endif()
endfunction()
