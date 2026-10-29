# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(_native_schema_dir
    ${CMAKE_CURRENT_BINARY_DIR}/include/executorch/backends/native/runtime
)
set(_native_schema_header ${_native_schema_dir}/native_graph_generated.h)
add_custom_command(
  OUTPUT ${_native_schema_header}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_native_schema_dir}
  COMMAND
    flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums -o
    ${_native_schema_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../serialization/native_graph.fbs
  DEPENDS flatc ${CMAKE_CURRENT_SOURCE_DIR}/../serialization/native_graph.fbs
  COMMENT "Generating native graph schema"
  VERBATIM
)
add_library(native_graph_schema INTERFACE ${_native_schema_header})
target_include_directories(
  native_graph_schema
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/flatbuffers/include>
)
add_custom_target(native_graph_schema_gen DEPENDS ${_native_schema_header})
add_dependencies(native_graph_schema native_graph_schema_gen)

add_library(
  native_graph
  graph/Argument.cpp
  graph/Graph.cpp
  graph/MemoryPlanning.cpp
  graph/Node.cpp
  graph/Scalar.cpp
  graph/TensorMeta.cpp
  graph/Value.cpp
  graph/utils/GraphUtils.cpp
)

add_library(native_reference_gguf_kquant utils/ReferenceGgufKQuant.cpp)
target_link_libraries(native_reference_gguf_kquant PRIVATE Threads::Threads)

add_library(
  native_package
  deserialize/OwnedBytes.cpp deserialize/Package.cpp
  deserialize/SafeTensorsReader.cpp deserialize/Samples.cpp
  deserialize/ZipReader.cpp
)
target_link_libraries(
  native_package
  PUBLIC native_graph
  PRIVATE libzip::zip nlohmann_json::nlohmann_json Threads::Threads
)

add_library(
  native_runtime Deserialize.cpp Program.cpp MethodMeta.cpp TensorInfo.cpp
                 Validation.cpp
)
target_link_libraries(
  native_runtime
  PUBLIC native_graph native_package
  PRIVATE $<BUILD_INTERFACE:native_graph_schema> Threads::Threads
)
add_dependencies(native_runtime native_graph_schema_gen)

add_library(native_engine engine/Engine.cpp)
target_link_libraries(native_engine PUBLIC native_runtime)

add_library(native_api api/Model.cpp)
target_link_libraries(native_api PUBLIC native_engine)

foreach(target native_graph native_reference_gguf_kquant native_package
               native_runtime native_engine native_api
)
  add_library(executorch::${target} ALIAS ${target})
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(
    ${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
                      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>
  )
  target_include_directories(
    ${target} PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
endforeach()

install(
  TARGETS native_graph native_reference_gguf_kquant native_package
          native_runtime native_engine native_api
  EXPORT ExecuTorchTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
