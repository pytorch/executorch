# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native MODULE pybindings.cpp)
target_link_libraries(_native PRIVATE executorch_native_sdk)
target_include_directories(_native PRIVATE "${EXECUTORCH_ROOT}/..")
set_target_properties(_native PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS _native LIBRARY DESTINATION . COMPONENT executorch_native)
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c [=[
from importlib.metadata import distribution

package = distribution("pybind11")
for path in package.files or ():
    if path.name.casefold().split(".")[0] in ("license", "copying"):
        license_path = package.locate_file(path)
        if license_path.is_file():
            print(license_path)
]=]
  OUTPUT_VARIABLE _pybind11_licenses
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_license_status
  ERROR_VARIABLE _pybind11_license_error
)
if(NOT _pybind11_license_status EQUAL 0 OR NOT _pybind11_licenses)
  message(
    FATAL_ERROR
      "Cannot locate pybind11 license files using ${Python_EXECUTABLE}. "
      "Reinstall pybind11 with its license files before building the Native wheel. "
      "${_pybind11_license_error}"
  )
endif()
string(REPLACE "\n" ";" _pybind11_licenses "${_pybind11_licenses}")
install(
  FILES ${_pybind11_licenses}
  DESTINATION share/licenses/pybind11
  COMPONENT executorch_native
)

install(
  FILES __init__.py __main__.py _native.pyi py.typed
  DESTINATION .
  COMPONENT executorch_native
)
