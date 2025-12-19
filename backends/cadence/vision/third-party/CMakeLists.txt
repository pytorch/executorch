cmake_minimum_required(VERSION 3.10.0)
project(cadence_vision)

# Collect all source files from the library directory
file(GLOB_RECURSE VISION_LIB_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/library/api/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/tables/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/dma.c"
)

# Create the vision library
add_library(xa_nnlib STATIC ${VISION_LIB_SOURCES})

# Set include directories
target_include_directories(xa_nnlib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(xa_nnlib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include_private
)

# Set properties for the library
set_target_properties(xa_nnlib PROPERTIES
  OUTPUT_NAME "xa_nnlib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

# Create output directories
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
