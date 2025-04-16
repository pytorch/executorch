# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Announce the name and value of a cmake variable in the summary of the build.
function(announce_configured_options NAME)
  get_property(configs GLOBAL PROPERTY _announce_configured_options)
  if(NOT configs)
    set_property(GLOBAL PROPERTY _announce_configured_options)
    get_property(configs GLOBAL PROPERTY _announce_configured_options)
  endif()

  set(option_exists FALSE)
  foreach(config IN LISTS configs)
    if(config STREQUAL "${NAME}")
      set(option_exists TRUE)
      break()
    endif()
  endforeach()

  if(NOT option_exists)
    set(configs ${configs} "${NAME}")
    set_property(GLOBAL PROPERTY _announce_configured_options "${configs}")
  endif()
endfunction()

# Print the configured options.
function(print_configured_options)
  get_property(configs GLOBAL PROPERTY _announce_configured_options)

  set(longest_name_length 0)
  foreach(config IN LISTS configs)
    string(LENGTH "${config}" length)
    if(length GREATER longest_name_length)
      set(longest_name_length ${length})
    endif()
  endforeach()

  message(STATUS "--- ⚙️ Configurations ---\n")
  foreach(config IN LISTS configs)
    string(LENGTH "${config}" config_length)
    math(EXPR num_spaces "${longest_name_length} - ${config_length}")
    set(padding "")
    while(num_spaces GREATER 0)
      set(padding "${padding} ")
      math(EXPR num_spaces "${num_spaces} - 1")
    endwhile()
    message(STATUS "${config}${padding} : ${${config}}")
  endforeach()
  message(STATUS "------------------------")
endfunction()
