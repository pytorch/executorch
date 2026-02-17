# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatbuffers")
  file(MAKE_DIRECTORY "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatbuffers")
endif()
file(MAKE_DIRECTORY
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src/build"
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep"
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/tmp"
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src/flatbuffers_ep-stamp"
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src"
  "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src/flatbuffers_ep-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src/flatbuffers_ep-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/sraut/ext_main/cad_rlc/executorch/third-party/flatc_ep/src/flatbuffers_ep-stamp${cfgdir}") # cfgdir has leading slash
endif()
