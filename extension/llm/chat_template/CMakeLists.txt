if(NOT EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER)
  return()
endif()

include(FetchContent)
cmake_policy(SET CMP0077 NEW)

FetchContent_Declare(
  jinja2cpp
  GIT_REPOSITORY https://github.com/jinja2cpp/Jinja2Cpp.git
  GIT_TAG 1.3.2
  GIT_SUBMODULES_RECURSE TRUE
)

set(JINJA2CPP_BUILD_TESTS
    OFF
    CACHE BOOL ""
          FORCE
)
set(JINJA2CPP_BUILD_SHARED
    OFF
    CACHE BOOL ""
          FORCE
)
set(JINJA2CPP_INSTALL
    OFF
    CACHE BOOL ""
          FORCE
)

FetchContent_MakeAvailable(jinja2cpp)
if(NOT TARGET jinja2cpp)
  message(FATAL_ERROR "Jinja2Cpp target not found after FetchContent.")
endif()

if(DEFINED jinja2cpp_SOURCE_DIR)
  function(executorch_copy_nonstd_header dep_name target header_name dest_root)
    set(_copied FALSE)
    if(TARGET ${target})
      get_target_property(_aliased ${target} ALIASED_TARGET)
      if(_aliased)
        set(_resolved_target ${_aliased})
      else()
        set(_resolved_target ${target})
      endif()
      get_target_property(
        _include_dirs ${_resolved_target} INTERFACE_INCLUDE_DIRECTORIES
      )
      foreach(_dir IN LISTS _include_dirs)
        if(EXISTS "${_dir}/nonstd/${header_name}")
          file(MAKE_DIRECTORY "${dest_root}/nonstd")
          file(
            COPY "${_dir}/nonstd/${header_name}"
            DESTINATION "${dest_root}/nonstd"
          )
          set(_copied TRUE)
          break()
        endif()
      endforeach()
    endif()
    if(NOT _copied)
      set(
        _fallback_path
        "${CMAKE_BINARY_DIR}/_deps/${dep_name}-src/include/nonstd/${header_name}"
      )
      if(EXISTS "${_fallback_path}")
        file(MAKE_DIRECTORY "${dest_root}/nonstd")
        file(COPY "${_fallback_path}" DESTINATION "${dest_root}/nonstd")
      endif()
    endif()
  endfunction()

  set(_jinja2cpp_nonstd_root
      "${jinja2cpp_SOURCE_DIR}/thirdparty/nonstd"
  )
  executorch_copy_nonstd_header(
    expected-lite
    nonstd::expected-lite
    expected.hpp
    "${_jinja2cpp_nonstd_root}/expected-lite/include"
  )
  executorch_copy_nonstd_header(
    variant-lite
    nonstd::variant-lite
    variant.hpp
    "${_jinja2cpp_nonstd_root}/variant-lite/include"
  )
  executorch_copy_nonstd_header(
    optional-lite
    nonstd::optional-lite
    optional.hpp
    "${_jinja2cpp_nonstd_root}/optional-lite/include"
  )
  executorch_copy_nonstd_header(
    string-view-lite
    nonstd::string-view-lite
    string_view.hpp
    "${_jinja2cpp_nonstd_root}/string-view-lite/include"
  )
endif()

set(SUPPORT_REGEX_LOOKAHEAD
    ON
    CACHE BOOL ""
          FORCE
)
