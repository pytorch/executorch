# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#
# Build tokenizers tests.
#
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

project(TokenizersTests)

# Include ExternalProject module
include(FindPackageHandleStandardArgs)
include(FetchContent)
include(ExternalProject)
FetchContent_Declare(
  tokenizers SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. BUILD_ALWAYS ON
)
set(SUPPORT_REGEX_LOOKAHEAD ON)
FetchContent_MakeAvailable(tokenizers)

# Build test
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)
FetchContent_MakeAvailable(googletest)

file(GLOB test_source_files test_*.cpp)

set(test_env "RESOURCES_PATH=${CMAKE_CURRENT_SOURCE_DIR}/resources")
enable_testing()
foreach(test_source_file ${test_source_files})
  get_filename_component(test_name ${test_source_file} NAME_WE)
  message(STATUS "Configuring unit test ${test_name}")
  add_executable(${test_name} ${test_source_file})
  target_include_directories(
    ${test_name} PRIVATE GTEST_INCLUDE_PATH ${TOKENIZERS_INCLUDE_DIRS}
  )
  target_link_libraries(${test_name} PUBLIC gtest_main GTest::gmock tokenizers)
  add_test(${test_name} "${test_name}")
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT ${test_env})
endforeach()
