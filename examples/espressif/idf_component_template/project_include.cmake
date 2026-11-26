# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ExecuTorch project API. ESP-IDF includes this file before processing any
# component, so a component can call these after its idf_component_register(),
# and the project after project():
#
# executorch_add_model(PTE <path.pte> [PTE ...])  # models this firmware runs
# executorch_add_kernels(DIR <bundle-dir>)        # from tools/gen_kernels.py
# executorch_add_backend(<BackendId>)             # force-link a delegate
#
# Calls are unioned across the project; a project has one kernel bundle. With
# models but no bundle, the build makes one if EXECUTORCH_PYTHON (-D or the
# environment; default: the build's Python) can import executorch.

function(executorch_add_model)
  cmake_parse_arguments(_arg "" "" "PTE" ${ARGN})
  if(NOT _arg_PTE OR _arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "usage: executorch_add_model(PTE <path.pte> [PTE ...])")
  endif()
  foreach(_path IN LISTS _arg_PTE)
    get_filename_component(
      _pte "${_path}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    idf_build_set_property(EXECUTORCH_MODELS "${_pte}" APPEND)
  endforeach()
endfunction()

function(executorch_add_kernels)
  cmake_parse_arguments(_arg "" "DIR" "" ${ARGN})
  if(NOT _arg_DIR OR _arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "usage: executorch_add_kernels(DIR <bundle-dir>)")
  endif()
  get_filename_component(
    _dir "${_arg_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  idf_build_get_property(_existing EXECUTORCH_KERNELS_DIR)
  if(_existing AND NOT _existing STREQUAL _dir)
    message(
      FATAL_ERROR
        "executorch_add_kernels: the project already uses the kernel bundle "
        "${_existing}; a project has exactly one. Make one bundle that "
        "covers all models."
    )
  endif()
  idf_build_set_property(EXECUTORCH_KERNELS_DIR "${_dir}")
endfunction()

function(executorch_add_backend backend)
  idf_build_set_property(EXECUTORCH_BACKENDS "${backend}" APPEND)
endfunction()

# Links the delegates named in ARGN into the component <lib>.
function(__executorch_link_backends lib)
  foreach(_backend IN LISTS ARGN)
    if(_backend STREQUAL "EspNNBackend")
      target_link_libraries(${lib} PRIVATE ${lib}_espnn)
      target_link_libraries(${lib} INTERFACE "-Wl,-u,et_anchor_backend_espnn")
    else()
      message(
        FATAL_ERROR
          "ExecuTorch: delegate ${_backend} is not provided by this component"
      )
    endif()
  endforeach()
endfunction()

# Deferred by the component to the end of the project's CMakeLists.txt, after
# every executorch_add_* call. Checks the bundle against the declared models and
# this component, then compiles its registration into the component.
function(__executorch_finalize)
  get_property(_lib GLOBAL PROPERTY EXECUTORCH_COMPONENT_LIB)
  get_property(_dir GLOBAL PROPERTY EXECUTORCH_COMPONENT_DIR)
  get_property(_version GLOBAL PROPERTY EXECUTORCH_COMPONENT_VERSION)
  get_property(_prim_src GLOBAL PROPERTY EXECUTORCH_PRIM_OPS_SRC)
  get_property(_options GLOBAL PROPERTY EXECUTORCH_SOURCE_OPTIONS)
  idf_build_get_property(_models EXECUTORCH_MODELS)
  idf_build_get_property(_bundle EXECUTORCH_KERNELS_DIR)
  idf_build_get_property(_backends EXECUTORCH_BACKENDS)

  if(NOT _models
     AND NOT _bundle
     AND NOT _backends
  )
    message(
      STATUS
        "ExecuTorch: no models or kernel bundle; nothing registered, and an "
        "app that calls et_model fails to link with an undefined "
        "et_requires_executorch_add_kernels"
    )
    return()
  endif()

  # et_model_create references this, so a project that runs models without
  # declaring them fails to link instead of failing every load.
  set(_out "${CMAKE_BINARY_DIR}/executorch")
  file(CONFIGURE OUTPUT "${_out}/declared.cpp" CONTENT
       "extern \"C\" const char et_requires_executorch_add_kernels = 0;\n"
  )
  target_sources(${_lib} PRIVATE "${_out}/declared.cpp")

  if(NOT _models AND NOT _bundle)
    list(REMOVE_DUPLICATES _backends)
    __executorch_link_backends(${_lib} ${_backends})
    if(CONFIG_ET_MAX_KERNEL_NUM)
      set(_registry ${CONFIG_ET_MAX_KERNEL_NUM})
    else()
      # Nothing registers kernels or prim ops, so the registry needs no room.
      set(_registry 1)
    endif()
    target_compile_definitions(${_lib} PRIVATE MAX_KERNEL_NUM=${_registry})
    message(STATUS "ExecuTorch: no kernel bundle; linking ${_backends}")
    return()
  endif()

  list(REMOVE_DUPLICATES _models)
  set(_command "python \"${_dir}/tools/gen_kernels.py\"")
  set(_pte_args "")
  set(_gen_args "")
  foreach(_pte IN LISTS _models)
    if(NOT EXISTS "${_pte}")
      message(FATAL_ERROR "ExecuTorch: declared model ${_pte} does not exist")
    endif()
    string(APPEND _pte_args " --pte \"${_pte}\"")
    list(APPEND _gen_args --pte "${_pte}")
  endforeach()
  if(NOT _bundle)
    if(NOT EXECUTORCH_PYTHON)
      set(EXECUTORCH_PYTHON "$ENV{EXECUTORCH_PYTHON}")
    endif()
    if(NOT EXECUTORCH_PYTHON)
      idf_build_get_property(EXECUTORCH_PYTHON PYTHON)
    elseif(NOT IS_ABSOLUTE "${EXECUTORCH_PYTHON}" OR NOT EXISTS
                                                     "${EXECUTORCH_PYTHON}"
    )
      message(
        FATAL_ERROR
          "ExecuTorch: EXECUTORCH_PYTHON must be the absolute path of a "
          "python, not ${EXECUTORCH_PYTHON}"
      )
    endif()
    string(
      CONCAT
        _manual
        "Make the bundle with the Python environment the models were exported "
        "with:\n  ${_command}${_pte_args} --out <bundle-dir>\nthen call "
        "executorch_add_kernels(DIR <bundle-dir>), or set EXECUTORCH_PYTHON to "
        "that environment's python."
    )
    execute_process(
      COMMAND "${EXECUTORCH_PYTHON}" -c
              "import executorch.codegen.tools.selective_build"
      RESULT_VARIABLE _no_executorch
      OUTPUT_QUIET
      ERROR_VARIABLE _probe_error
    )
    if(_no_executorch)
      message(
        FATAL_ERROR
          "ExecuTorch: models are declared but no kernel bundle is, and "
          "${EXECUTORCH_PYTHON} cannot import executorch to make one:\n"
          "${_probe_error}${_manual}"
      )
    endif()
    set(_bundle "${CMAKE_BINARY_DIR}/executorch_kernels")
    message(
      STATUS "ExecuTorch: making the kernel bundle with ${EXECUTORCH_PYTHON}"
    )
    execute_process(
      COMMAND "${EXECUTORCH_PYTHON}" "${_dir}/tools/gen_kernels.py" ${_gen_args}
              --out "${_bundle}"
      RESULT_VARIABLE _failed
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _output
    )
    if(_failed)
      message(
        FATAL_ERROR
          "ExecuTorch: ${EXECUTORCH_PYTHON} failed to make the kernel bundle:\n"
          "${_output}${_manual}"
      )
    endif()
  endif()

  set(_manifest "${_bundle}/kernels.json")
  if(NOT EXISTS "${_manifest}")
    message(
      FATAL_ERROR
        "ExecuTorch: ${_bundle} is not a kernel bundle (no kernels.json)"
    )
  endif()
  file(READ "${_manifest}" _json)
  string(
    JSON
    _schema
    ERROR_VARIABLE
    _error
    GET
    "${_json}"
    schema
  )
  if(_error OR NOT _schema EQUAL 1)
    message(
      FATAL_ERROR "ExecuTorch: ${_manifest} is not a schema 1 kernel bundle"
    )
  endif()
  set(_recorded "")
  set(_recorded_names "")
  string(JSON _n_models ERROR_VARIABLE _error LENGTH "${_json}" models)
  if(NOT _error AND _n_models GREATER 0)
    math(EXPR _last "${_n_models} - 1")
    foreach(_i RANGE ${_last})
      string(JSON _sha GET "${_json}" models ${_i} sha256)
      string(
        JSON
        _name
        GET
        "${_json}"
        models
        ${_i}
        name
      )
      list(APPEND _recorded ${_sha})
      list(APPEND _recorded_names ${_name})
    endforeach()
  endif()

  if(_models)
    string(APPEND _command "${_pte_args}")
  else()
    foreach(_name IN LISTS _recorded_names)
      string(APPEND _command " --pte <path to ${_name}>")
    endforeach()
  endif()
  string(JSON _n_ops ERROR_VARIABLE _error LENGTH "${_json}" ops)
  if(NOT _error AND _n_ops GREATER 0)
    math(EXPR _last "${_n_ops} - 1")
    set(_ops "")
    foreach(_i RANGE ${_last})
      string(JSON _op GET "${_json}" ops ${_i})
      list(APPEND _ops ${_op})
    endforeach()
    list(JOIN _ops "," _ops)
    string(APPEND _command " --ops ${_ops}")
  endif()
  string(APPEND _command " --out \"${_bundle}\"")
  string(CONCAT _regenerate "Regenerate it with the Python environment the "
                "models were exported with:\n  ${_command}"
  )

  string(
    JSON
    _bundle_version
    ERROR_VARIABLE
    _error
    GET
    "${_json}"
    component_version
  )
  if(NOT _bundle_version STREQUAL _version)
    message(
      FATAL_ERROR
        "ExecuTorch: kernel bundle ${_bundle} was made for component_version "
        "'${_bundle_version}', but this component is ${_version}. "
        "${_regenerate}"
    )
  endif()
  # Development builds from different commits can share a version.
  string(
    JSON
    _bundle_commit
    ERROR_VARIABLE
    _error
    GET
    "${_json}"
    component_commit
  )
  file(STRINGS "${_dir}/VERSION" _commit REGEX "^executorch_commit:")
  string(REGEX REPLACE "^executorch_commit:[ ]*([^ ]*).*" "\\1" _commit
                       "${_commit}"
  )
  if(NOT _bundle_commit STREQUAL _commit)
    message(
      FATAL_ERROR
        "ExecuTorch: kernel bundle ${_bundle} was made for component_commit "
        "'${_bundle_commit}', but this component is from ${_commit}. "
        "${_regenerate}"
    )
  endif()

  if(_models)
    set(_accounted "")
    set(_problems "")
    foreach(_pte IN LISTS _models)
      file(SHA256 "${_pte}" _sha)
      if(_sha IN_LIST _recorded)
        list(APPEND _accounted ${_sha})
        continue()
      endif()
      get_filename_component(_pte_name "${_pte}" NAME)
      set(_problem "not in the bundle")
      list(FIND _recorded_names "${_pte_name}" _i)
      if(_i GREATER_EQUAL 0)
        list(GET _recorded ${_i} _old)
        if(NOT _old IN_LIST _accounted)
          list(APPEND _accounted ${_old})
          set(_problem "changed since the bundle was made")
        endif()
      endif()
      list(APPEND _problems "  ${_pte}: ${_problem}")
    endforeach()
    foreach(_sha _name IN ZIP_LISTS _recorded _recorded_names)
      if(NOT _sha IN_LIST _accounted)
        list(APPEND _problems "  ${_name}: in the bundle, not declared")
      endif()
    endforeach()
    if(_problems)
      list(JOIN _problems "\n" _problems)
      message(
        FATAL_ERROR
          "ExecuTorch: kernel bundle ${_bundle} is stale for the declared "
          "models:\n${_problems}\n${_regenerate}"
      )
    endif()
  elseif(_recorded_names)
    list(JOIN _recorded_names ", " _names)
    message(
      WARNING
        "ExecuTorch: no models declared, so ${_bundle} is used as is and is "
        "not checked against ${_names}, which it was made from. Declare them "
        "with executorch_add_model() to catch a re-export at configure time."
    )
  else()
    message(STATUS "ExecuTorch: no models declared; using ${_bundle} as is")
  endif()

  string(JSON _n_backends LENGTH "${_json}" backends)
  if(_n_backends GREATER 0)
    math(EXPR _last "${_n_backends} - 1")
    foreach(_i RANGE ${_last})
      string(JSON _backend GET "${_json}" backends ${_i})
      list(APPEND _backends ${_backend})
    endforeach()
  endif()
  if(_backends)
    list(REMOVE_DUPLICATES _backends)
    __executorch_link_backends(${_lib} ${_backends})
  endif()

  set(_sources "")
  set(_anchors et_anchor_prim_ops)
  set(_summary "")
  string(JSON _n_libraries LENGTH "${_json}" libraries)
  if(_n_libraries GREATER 0)
    math(EXPR _last "${_n_libraries} - 1")
    foreach(_i RANGE ${_last})
      string(JSON _name MEMBER "${_json}" libraries ${_i})
      set(_registration
          "${_bundle}/${_name}/RegisterCodegenUnboxedKernelsEverything.cpp"
      )
      if(NOT _name MATCHES "^[A-Za-z_][A-Za-z0-9_]*$" OR NOT EXISTS
                                                         "${_registration}"
      )
        message(
          FATAL_ERROR
            "ExecuTorch: kernel bundle ${_bundle} is incomplete; library "
            "'${_name}' has no registration. ${_regenerate}"
        )
      endif()
      # The anchor lives in a wrapper because codegen's registration has only
      # file-local symbols for -u to name.
      file(
        CONFIGURE
        OUTPUT
        "${_out}/register_kernels_${_name}.cpp"
        CONTENT
        "#include \"${_registration}\"\nextern \"C\" const char et_anchor_kernels_${_name} = 0;\n"
      )
      list(APPEND _sources "${_out}/register_kernels_${_name}.cpp")
      list(APPEND _anchors et_anchor_kernels_${_name})
      string(JSON _n_ops LENGTH "${_json}" libraries ${_name})
      list(APPEND _summary "${_name} (${_n_ops} ops)")
    endforeach()
  endif()
  file(
    CONFIGURE
    OUTPUT
    "${_out}/register_prim_ops.cpp"
    CONTENT
    "#include \"${_dir}/${_prim_src}\"\nextern \"C\" const char et_anchor_prim_ops = 0;\n"
  )
  set_source_files_properties(
    "${_out}/register_prim_ops.cpp" TARGET_DIRECTORY ${_lib}
    PROPERTIES
      COMPILE_DEFINITIONS
      "ET_PRIM_OPS_SELECTIVE_BUILD;EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD"
      INCLUDE_DIRECTORIES "${_bundle}"
  )
  list(APPEND _sources "${_out}/register_prim_ops.cpp")
  set_source_files_properties(
    ${_sources} TARGET_DIRECTORY ${_lib} PROPERTIES COMPILE_OPTIONS
                                                    "${_options}"
  )

  target_sources(${_lib} PRIVATE ${_sources})
  # Prim op registration uses et_view and et_copy_index from the kernels.
  string(JSON _n_prims ERROR_VARIABLE _error LENGTH "${_json}" prim_ops)
  if(_n_libraries GREATER 0 OR _n_prims GREATER 0)
    target_link_libraries(${_lib} PRIVATE ${_lib}_kernels)
  endif()
  set(_kernel_num_h
      "${_bundle}/include/executorch/runtime/kernel/selected_max_kernel_num.h"
  )
  set(_kernel_num "")
  if(EXISTS "${_kernel_num_h}")
    file(STRINGS "${_kernel_num_h}" _kernel_num
         REGEX "EXECUTORCH_SELECTED_MAX_KERNEL_NUM [0-9]+"
    )
    string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _kernel_num "${_kernel_num}")
  endif()
  if(NOT _kernel_num MATCHES "^[0-9]+$")
    message(
      FATAL_ERROR
        "ExecuTorch: kernel bundle ${_bundle} is incomplete; it has no "
        "kernel count in ${_kernel_num_h}. ${_regenerate}"
    )
  endif()
  if(NOT CONFIG_ET_MAX_KERNEL_NUM)
    set(_registry ${_kernel_num})
  elseif(CONFIG_ET_MAX_KERNEL_NUM LESS _kernel_num)
    message(
      FATAL_ERROR
        "ExecuTorch: CONFIG_ET_MAX_KERNEL_NUM=${CONFIG_ET_MAX_KERNEL_NUM} is "
        "below the ${_kernel_num} kernels of ${_bundle}. Set it in sdkconfig "
        "to at least ${_kernel_num}, or to 0 to size the registry from the "
        "bundle."
    )
  else()
    set(_registry ${CONFIG_ET_MAX_KERNEL_NUM})
  endif()
  target_compile_definitions(${_lib} PRIVATE MAX_KERNEL_NUM=${_registry})
  foreach(_anchor IN LISTS _anchors)
    target_link_libraries(${_lib} INTERFACE "-Wl,-u,${_anchor}")
  endforeach()
  set_property(
    DIRECTORY "${CMAKE_SOURCE_DIR}"
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest}" "${_dir}/VERSION"
             ${_models}
  )
  list(JOIN _summary ", " _summary)
  if(NOT _summary)
    set(_summary "no kernel libraries")
  endif()
  message(
    STATUS
      "ExecuTorch: kernel bundle ${_bundle}: ${_summary}; ${_registry} registry slots"
  )
endfunction()
