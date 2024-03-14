# Copyright © 2023 Apple Inc. All rights reserved.

cmake_minimum_required(VERSION 3.19)

project(executorch_coreml_backend)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

# inmemoryfs sources
set(INMEMORYFS_SOURCES
  runtime/inmemoryfs/inmemory_filesystem.cpp
  runtime/inmemoryfs/inmemory_filesystem_utils.mm
  runtime/inmemoryfs/memory_buffer.cpp
  runtime/inmemoryfs/memory_stream.cpp
  runtime/inmemoryfs/reversed_memory_stream.cpp
)

# kvstore sources
set(KVSTORE_SOURCES
  runtime/kvstore/database.cpp
  runtime/kvstore/json_key_value_store.cpp
  runtime/kvstore/sqlite_error.cpp
  runtime/kvstore/key_value_store.cpp
  runtime/kvstore/statement.cpp
)

# delegate sources
set(DELEGATE_SOURCES
  runtime/delegate/asset.mm
  runtime/delegate/backend_delegate.mm
  runtime/delegate/coreml_backend_delegate.mm
  runtime/delegate/ETCoreMLAsset.mm
  runtime/delegate/ETCoreMLAssetManager.mm
  runtime/delegate/ETCoreMLLogging.mm
  runtime/delegate/ETCoreMLModel.mm
  runtime/delegate/ETCoreMLModelManager.mm
  runtime/delegate/ETCoreMLStrings.mm
  runtime/delegate/MLModel_Prewarm.mm
  runtime/delegate/MLMultiArray_Copy.mm
  runtime/delegate/multiarray.mm
  runtime/delegate/serde_json.mm
)

# util sources
set(UTIL_SOURCES
  runtime/util/json_util.cpp
  runtime/util/objc_json_serde.mm
)

# Define the delegate library
add_library(coremldelegate)
target_sources(
  coremldelegate PRIVATE
  ${INMEMORYFS_SOURCES}
  ${KVSTORE_SOURCES}
  ${DELEGATE_SOURCES}
  ${UTIL_SOURCES}
)
target_include_directories(
  coremldelegate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_include_directories(
  coremldelegate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kvstore
)
target_include_directories(
  coremldelegate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/inmemoryfs
)
target_include_directories(
  coremldelegate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/delegate
)
target_include_directories(
  coremldelegate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/util
)
target_include_directories(
  coremldelegate PRIVATE
  ${EXECUTORCH_ROOT}/..
)
target_compile_options(executorch PUBLIC -DET_EVENT_TRACER_ENABLED)

find_library(ACCELERATE_FRAMEWORK Accelerate)
find_library(COREML_FRAMEWORK CoreML)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(SQLITE_LIBRARY sqlite3)

target_link_libraries(coremldelegate
  PRIVATE
  executorch
  ${ACCELERATE_FRAMEWORK}
  ${COREML_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${SQLITE_LIBRARY}
)

target_compile_options(coremldelegate PRIVATE "-fobjc-arc")
target_compile_options(coremldelegate PRIVATE "-fno-exceptions")
target_compile_options(coremldelegate PRIVATE "-fno-rtti")

set(
  TARGET coremldelegate
  APPEND_STRING PROPERTY COMPILE_FLAGS "-x objective-c++"
)

set(
  TARGET coremldelegate
  APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-null-character"
)

set(
  TARGET coremldelegate
  APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-receiver-expr"
)
