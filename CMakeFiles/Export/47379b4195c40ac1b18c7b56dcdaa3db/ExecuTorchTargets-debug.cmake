#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cpuinfo" for configuration "Debug"
set_property(TARGET cpuinfo APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cpuinfo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libcpuinfo.a"
  )

list(APPEND _cmake_import_check_targets cpuinfo )
list(APPEND _cmake_import_check_files_for_cpuinfo "${_IMPORT_PREFIX}/lib64/libcpuinfo.a" )

# Import target "pthreadpool" for configuration "Debug"
set_property(TARGET pthreadpool APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(pthreadpool PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libpthreadpool.a"
  )

list(APPEND _cmake_import_check_targets pthreadpool )
list(APPEND _cmake_import_check_files_for_pthreadpool "${_IMPORT_PREFIX}/lib64/libpthreadpool.a" )

# Import target "kernels_util_all_deps" for configuration "Debug"
set_property(TARGET kernels_util_all_deps APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kernels_util_all_deps PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libkernels_util_all_deps.a"
  )

list(APPEND _cmake_import_check_targets kernels_util_all_deps )
list(APPEND _cmake_import_check_files_for_kernels_util_all_deps "${_IMPORT_PREFIX}/lib64/libkernels_util_all_deps.a" )

# Import target "portable_kernels" for configuration "Debug"
set_property(TARGET portable_kernels APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(portable_kernels PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libportable_kernels.a"
  )

list(APPEND _cmake_import_check_targets portable_kernels )
list(APPEND _cmake_import_check_files_for_portable_kernels "${_IMPORT_PREFIX}/lib64/libportable_kernels.a" )

# Import target "portable_ops_lib" for configuration "Debug"
set_property(TARGET portable_ops_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(portable_ops_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libportable_ops_lib.a"
  )

list(APPEND _cmake_import_check_targets portable_ops_lib )
list(APPEND _cmake_import_check_files_for_portable_ops_lib "${_IMPORT_PREFIX}/lib64/libportable_ops_lib.a" )

# Import target "executorch" for configuration "Debug"
set_property(TARGET executorch APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(executorch PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libexecutorch.a"
  )

list(APPEND _cmake_import_check_targets executorch )
list(APPEND _cmake_import_check_files_for_executorch "${_IMPORT_PREFIX}/lib64/libexecutorch.a" )

# Import target "executorch_core" for configuration "Debug"
set_property(TARGET executorch_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(executorch_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libexecutorch_core.a"
  )

list(APPEND _cmake_import_check_targets executorch_core )
list(APPEND _cmake_import_check_files_for_executorch_core "${_IMPORT_PREFIX}/lib64/libexecutorch_core.a" )

# Import target "extension_data_loader" for configuration "Debug"
set_property(TARGET extension_data_loader APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(extension_data_loader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libextension_data_loader.a"
  )

list(APPEND _cmake_import_check_targets extension_data_loader )
list(APPEND _cmake_import_check_files_for_extension_data_loader "${_IMPORT_PREFIX}/lib64/libextension_data_loader.a" )

# Import target "extension_evalue_util" for configuration "Debug"
set_property(TARGET extension_evalue_util APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(extension_evalue_util PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/sraut/ext_main/cad_rlc/executorch/lib/libextension_evalue_util.a"
  )

list(APPEND _cmake_import_check_targets extension_evalue_util )
list(APPEND _cmake_import_check_files_for_extension_evalue_util "/home/sraut/ext_main/cad_rlc/executorch/lib/libextension_evalue_util.a" )

# Import target "extension_flat_tensor" for configuration "Debug"
set_property(TARGET extension_flat_tensor APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(extension_flat_tensor PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libextension_flat_tensor.a"
  )

list(APPEND _cmake_import_check_targets extension_flat_tensor )
list(APPEND _cmake_import_check_files_for_extension_flat_tensor "${_IMPORT_PREFIX}/lib64/libextension_flat_tensor.a" )

# Import target "extension_runner_util" for configuration "Debug"
set_property(TARGET extension_runner_util APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(extension_runner_util PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libextension_runner_util.a"
  )

list(APPEND _cmake_import_check_targets extension_runner_util )
list(APPEND _cmake_import_check_files_for_extension_runner_util "${_IMPORT_PREFIX}/lib64/libextension_runner_util.a" )

# Import target "extension_threadpool" for configuration "Debug"
set_property(TARGET extension_threadpool APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(extension_threadpool PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/libextension_threadpool.a"
  )

list(APPEND _cmake_import_check_targets extension_threadpool )
list(APPEND _cmake_import_check_files_for_extension_threadpool "${_IMPORT_PREFIX}/lib64/libextension_threadpool.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
