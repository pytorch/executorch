CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project
PROJECT(FP16 C CXX)

# ---[ Options.
OPTION(FP16_BUILD_TESTS "Build FP16 unit tests" ON)
OPTION(FP16_BUILD_BENCHMARKS "Build FP16 micro-benchmarks" ON)

# ---[ CMake options
IF(FP16_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

MACRO(FP16_TARGET_ENABLE_CXX11 target)
  IF(${CMAKE_VERSION} VERSION_LESS "3.1")
    IF(NOT MSVC)
      TARGET_COMPILE_OPTIONS(${target} PRIVATE -std=c++11)
    ENDIF()
  ELSE()
    SET_TARGET_PROPERTIES(${target} PROPERTIES
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS YES)
  ENDIF()
ENDMACRO()

# ---[ Download deps
IF(NOT DEFINED PSIMD_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PSimd to ${CMAKE_BINARY_DIR}/psimd-source (define PSIMD_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CMAKE_BINARY_DIR}/psimd-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/psimd-download")
  SET(PSIMD_SOURCE_DIR "${CMAKE_BINARY_DIR}/psimd-source" CACHE STRING "PSimd source directory")
ENDIF()

IF(FP16_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Test to ${CMAKE_BINARY_DIR}/googletest-source (define GOOGLETEST_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  SET(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source" CACHE STRING "Google Test source directory")
ENDIF()

IF(FP16_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
  SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
ENDIF()

# ---[ FP16 library
IF(${CMAKE_VERSION} VERSION_LESS "3.0")
  ADD_LIBRARY(fp16 STATIC
    include/fp16.h
    include/fp16/fp16.h
    include/fp16/bitcasts.h
    include/fp16/psimd.h)
  SET_TARGET_PROPERTIES(fp16 PROPERTIES LINKER_LANGUAGE C)
ELSE()
  ADD_LIBRARY(fp16 INTERFACE)
ENDIF()
TARGET_INCLUDE_DIRECTORIES(fp16 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

INSTALL(FILES include/fp16.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(FILES
    include/fp16/bitcasts.h
    include/fp16/fp16.h
    include/fp16/psimd.h
    include/fp16/__init__.py
    include/fp16/avx.py
    include/fp16/avx2.py
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fp16)

# ---[ Configure psimd
IF(NOT TARGET psimd)
  ADD_SUBDIRECTORY(
    "${PSIMD_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/psimd")
ENDIF()

IF(FP16_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googletest")
  ENDIF()

  # ---[ Build FP16 unit tests
  ADD_EXECUTABLE(ieee-to-fp32-bits-test test/ieee-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-bits-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-to-fp32-bits-test fp16 gtest gtest_main)
  ADD_TEST(ieee-to-fp32-bits ieee-to-fp32-bits-test)

  ADD_EXECUTABLE(ieee-to-fp32-value-test test/ieee-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-to-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-to-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(ieee-to-fp32-value ieee-to-fp32-value-test)

  ADD_EXECUTABLE(ieee-from-fp32-value-test test/ieee-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(ieee-from-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(ieee-from-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(ieee-from-fp32-value ieee-from-fp32-value-test)

  ADD_EXECUTABLE(alt-to-fp32-bits-test test/alt-to-fp32-bits.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-bits-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-to-fp32-bits-test fp16 gtest gtest_main)
  ADD_TEST(alt-to-fp32-bits alt-to-fp32-bits-test)

  ADD_EXECUTABLE(alt-to-fp32-value-test test/alt-to-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-to-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-to-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(alt-to-fp32-value alt-to-fp32-value-test)

  ADD_EXECUTABLE(alt-from-fp32-value-test test/alt-from-fp32-value.cc test/tables.cc)
  TARGET_INCLUDE_DIRECTORIES(alt-from-fp32-value-test PRIVATE test)
  TARGET_LINK_LIBRARIES(alt-from-fp32-value-test fp16 gtest gtest_main)
  ADD_TEST(alt-from-fp32-value alt-from-fp32-value-test)

  ADD_EXECUTABLE(bitcasts-test test/bitcasts.cc)
  TARGET_LINK_LIBRARIES(bitcasts-test fp16 gtest gtest_main)
  ADD_TEST(bitcasts bitcasts-test)
ENDIF()

IF(FP16_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/googlebenchmark")
  ENDIF()

  # ---[ Build FP16 benchmarks
  ADD_EXECUTABLE(ieee-element-bench bench/ieee-element.cc)
  TARGET_COMPILE_DEFINITIONS(ieee-element-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(ieee-element-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(ieee-element-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(alt-element-bench bench/alt-element.cc)
  TARGET_LINK_LIBRARIES(alt-element-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(from-ieee-array-bench bench/from-ieee-array.cc)
  FP16_TARGET_ENABLE_CXX11(from-ieee-array-bench)
  TARGET_COMPILE_DEFINITIONS(from-ieee-array-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(from-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(from-ieee-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(from-alt-array-bench bench/from-alt-array.cc)
  FP16_TARGET_ENABLE_CXX11(from-alt-array-bench)
  TARGET_LINK_LIBRARIES(from-alt-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(to-ieee-array-bench bench/to-ieee-array.cc)
  FP16_TARGET_ENABLE_CXX11(to-ieee-array-bench)
  TARGET_COMPILE_DEFINITIONS(to-ieee-array-bench PRIVATE FP16_COMPARATIVE_BENCHMARKS=1)
  TARGET_INCLUDE_DIRECTORIES(to-ieee-array-bench PRIVATE ${PROJECT_SOURCE_DIR})
  TARGET_LINK_LIBRARIES(to-ieee-array-bench fp16 psimd benchmark)

  ADD_EXECUTABLE(to-alt-array-bench bench/to-alt-array.cc)
  FP16_TARGET_ENABLE_CXX11(to-alt-array-bench)
  TARGET_LINK_LIBRARIES(to-alt-array-bench fp16 psimd benchmark)
ENDIF()
