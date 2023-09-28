# Copyright © 2023 Apple Inc. All rights reserved.

cmake_minimum_required(VERSION 3.19)

project(executorch_coreml_backend)

enable_language(CXX)
enable_language(OBJC)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# inmemoryfs sources
set(INMEMORYFS_SOURCES
  runtime/inmemoryfs/inmemory_filesystem.cpp
  runtime/inmemoryfs/memory_buffer.cpp
  runtime/inmemoryfs/memory_stream.cpp
  runtime/inmemoryfs/reversed_memory_stream.cpp
)

# kvstore sources
set(KVSTORE_SOURCES
  runtime/kvstore/database.cpp
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

# Define the delegate library
add_library(coremldelegate)
target_sources(
  coremldelegate PRIVATE
  ${INMEMORYFS_SOURCES}
  ${KVSTORE_SOURCES}
  ${DELEGATE_SOURCES}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/third-party/nlohmann
)
target_include_directories(
  coremldelegate PRIVATE
  ${EXECUTORCH_ROOT}/..
)

target_link_libraries(
  coremldelegate PRIVATE
  executorch
  gflags
)

target_compile_options(coremldelegate PUBLIC "-fobjc-arc")

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

set_property(
  TARGET coremldelegate
  PROPERTY CXX_STANDARD 17
)

