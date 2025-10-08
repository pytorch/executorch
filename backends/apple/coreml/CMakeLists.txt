# Copyright © 2023 Apple Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(APPLE)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  find_library(COREML_FRAMEWORK CoreML REQUIRED)
  find_library(SQLITE_LIBRARY sqlite3 REQUIRED)
endif()

if(EXECUTORCH_BUILD_DEVTOOLS)
  # protobuf requires frtti
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -frtti")
endif()

# kvstore sources
set(KVSTORE_SOURCES
    runtime/kvstore/database.cpp runtime/kvstore/json_key_value_store.cpp
    runtime/kvstore/sqlite_error.cpp runtime/kvstore/key_value_store.cpp
    runtime/kvstore/statement.cpp
)

# delegate sources
set(DELEGATE_SOURCES
    runtime/delegate/asset.mm
    runtime/delegate/backend_delegate.mm
    runtime/delegate/coreml_backend_delegate.mm
    runtime/delegate/ETCoreMLAsset.mm
    runtime/delegate/ETCoreMLAssetManager.mm
    runtime/delegate/ETCoreMLDefaultModelExecutor.mm
    runtime/delegate/ETCoreMLModelLoader.mm
    runtime/delegate/ETCoreMLModelCompiler.mm
    runtime/delegate/ETCoreMLLogging.mm
    runtime/delegate/ETCoreMLModel.mm
    runtime/delegate/ETCoreMLModelManager.mm
    runtime/delegate/ETCoreMLStrings.mm
    runtime/delegate/MLModel_Prewarm.mm
    runtime/delegate/MLMultiArray_Copy.mm
    runtime/delegate/multiarray.mm
    runtime/delegate/serde_json.mm
)

# sdk sources
set(SDK_SOURCES
    runtime/sdk/ETCoreMLModelAnalyzer.mm
    runtime/sdk/ETCoreMLModelStructurePath.mm
    runtime/sdk/ETCoreMLOperationProfilingInfo.mm
    runtime/sdk/ETCoreMLModelDebugInfo.mm
    runtime/sdk/ETCoreMLModelDebugger.mm
    runtime/sdk/ETCoreMLModelProfiler.mm
    runtime/sdk/ETCoreMLPair.mm
    runtime/sdk/model_package_info.mm
    runtime/sdk/model_event_logger_impl.mm
    runtime/sdk/program_path.mm
)

# protobuf sources
set(PROTOBUF_SOURCES
    runtime/sdk/format/ArrayFeatureExtractor.pb.cc
    runtime/sdk/format/AudioFeaturePrint.pb.cc
    runtime/sdk/format/BayesianProbitRegressor.pb.cc
    runtime/sdk/format/CategoricalMapping.pb.cc
    runtime/sdk/format/ClassConfidenceThresholding.pb.cc
    runtime/sdk/format/CustomModel.pb.cc
    runtime/sdk/format/DataStructures.pb.cc
    runtime/sdk/format/DictVectorizer.pb.cc
    runtime/sdk/format/FeatureTypes.pb.cc
    runtime/sdk/format/FeatureVectorizer.pb.cc
    runtime/sdk/format/Gazetteer.pb.cc
    runtime/sdk/format/GLMClassifier.pb.cc
    runtime/sdk/format/GLMRegressor.pb.cc
    runtime/sdk/format/Identity.pb.cc
    runtime/sdk/format/Imputer.pb.cc
    runtime/sdk/format/ItemSimilarityRecommender.pb.cc
    runtime/sdk/format/LinkedModel.pb.cc
    runtime/sdk/format/MIL.pb.cc
    runtime/sdk/format/Model.pb.cc
    runtime/sdk/format/NearestNeighbors.pb.cc
    runtime/sdk/format/NeuralNetwork.pb.cc
    runtime/sdk/format/NonMaximumSuppression.pb.cc
    runtime/sdk/format/Normalizer.pb.cc
    runtime/sdk/format/NonMaximumSuppression.pb.cc
    runtime/sdk/format/OneHotEncoder.pb.cc
    runtime/sdk/format/Parameters.pb.cc
    runtime/sdk/format/Scaler.pb.cc
    runtime/sdk/format/SoundAnalysisPreprocessing.pb.cc
    runtime/sdk/format/SVM.pb.cc
    runtime/sdk/format/TextClassifier.pb.cc
    runtime/sdk/format/TreeEnsemble.pb.cc
    runtime/sdk/format/VisionFeaturePrint.pb.cc
    runtime/sdk/format/WordEmbedding.pb.cc
    runtime/sdk/format/WordTagger.pb.cc
)

# CoreML util

set(UTIL_SOURCES runtime/util/json_util.cpp)
if(APPLE)
  list(APPEND UTIL_SOURCES runtime/util/objc_json_serde.mm)
endif()

add_library(coreml_util ${UTIL_SOURCES})
target_include_directories(
  coreml_util
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/util>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/apple/coreml/runtime/util>
)
if(APPLE)
  target_link_libraries(coreml_util PRIVATE ${FOUNDATION_FRAMEWORK})
endif()
target_compile_options(coreml_util PUBLIC -fPIC)

install(TARGETS coreml_util DESTINATION lib)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/runtime/util
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/apple/coreml/runtime/util
  FILES_MATCHING
  PATTERN "*.h"
)

# CoreML inmemoryfs

set(INMEMORYFS_SOURCES
    runtime/inmemoryfs/inmemory_filesystem.cpp
    runtime/inmemoryfs/memory_buffer.cpp runtime/inmemoryfs/memory_stream.cpp
    runtime/inmemoryfs/reversed_memory_stream.cpp
)
if(APPLE)
  list(APPEND INMEMORYFS_SOURCES
       runtime/inmemoryfs/inmemory_filesystem_utils.mm
  )
endif()

add_library(coreml_inmemoryfs ${INMEMORYFS_SOURCES})
target_include_directories(
  coreml_inmemoryfs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/inmemoryfs>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/apple/coreml/runtime/inmemoryfs>
)
if(APPLE)
  target_link_libraries(
    coreml_inmemoryfs PRIVATE coreml_util ${FOUNDATION_FRAMEWORK}
  )
endif()
target_compile_options(coreml_inmemoryfs PUBLIC -fPIC)

install(
  TARGETS coreml_inmemoryfs
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/runtime/inmemoryfs
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/apple/coreml/runtime/inmemoryfs
  FILES_MATCHING
  PATTERN "*.h"
)

# executorchcoreml

if(EXECUTORCH_BUILD_PYBIND)
  pybind11_add_module(
    executorchcoreml SHARED runtime/inmemoryfs/inmemory_filesystem_py.cpp
    runtime/inmemoryfs/inmemory_filesystem_utils.cpp
  )
  target_link_libraries(
    executorchcoreml PRIVATE coreml_util coreml_inmemoryfs
                             nlohmann_json::nlohmann_json
  )
  target_compile_options(executorchcoreml PUBLIC -fPIC)
endif()

# coremldelegate

if(APPLE)
  add_library(coremldelegate)
  target_sources(coremldelegate PRIVATE ${KVSTORE_SOURCES} ${DELEGATE_SOURCES})

  target_include_directories(
    coremldelegate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_include_directories(
    coremldelegate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kvstore
  )
  target_include_directories(
    coremldelegate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/delegate
  )
  target_include_directories(coremldelegate PRIVATE ${PROJECT_SOURCE_DIR}/..)
  target_include_directories(
    coremldelegate PRIVATE ${PROJECT_SOURCE_DIR}/runtime/core/portable_type/c10
  )
  target_compile_definitions(
    coremldelegate PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
  )

  if(EXECUTORCH_BUILD_DEVTOOLS)
    target_sources(coremldelegate PRIVATE ${SDK_SOURCES} ${PROTOBUF_SOURCES})
    target_include_directories(
      coremldelegate
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/sdk
        ${CMAKE_CURRENT_SOURCE_DIR}/third-party/coremltools/deps/protobuf/src
    )
    add_subdirectory(
      ${CMAKE_CURRENT_SOURCE_DIR}/third-party/coremltools/deps/protobuf/cmake
    )

    executorch_target_link_options_shared_lib(libprotobuf-lite)
    target_link_libraries(coremldelegate PRIVATE libprotobuf-lite)
  endif()

  target_link_libraries(
    coremldelegate
    PUBLIC coreml_util coreml_inmemoryfs
    PRIVATE executorch_core ${ACCELERATE_FRAMEWORK} ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK} ${SQLITE_LIBRARY}
  )

  executorch_target_link_options_shared_lib(coremldelegate)

  if(EXECUTORCH_COREML_BUILD_EXECUTOR_RUNNER)
    target_link_libraries(
      coremldelegate PRIVATE portable_ops_lib portable_kernels
    )
  endif()

  target_compile_options(
    coremldelegate PRIVATE -fobjc-arc -fno-exceptions -x objective-c++
                           -Wno-null-character -Wno-receiver-expr
  )

  if(EXECUTORCH_BUILD_DEVTOOLS)
    target_compile_options(coremldelegate PRIVATE "-frtti")
    target_compile_options(libprotobuf-lite PRIVATE "-frtti")
  else()
    target_compile_options(coremldelegate PRIVATE "-fno-rtti")
  endif()

  install(
    TARGETS coremldelegate coreml_util coreml_inmemoryfs
    EXPORT ExecuTorchTargets
    DESTINATION lib
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()
