cmake_minimum_required(VERSION 3.10.0)
project(cadence_vision)

# Collect all source files from the library directory
file(GLOB_RECURSE VISION_LIB_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/library/api/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/tables/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/dma.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/memory_manager.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/library/utils.c"
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

# ============================================================================
# libxai_common - Common utilities and data types
# ============================================================================
set(LIBXAI_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai_common/src/xai_buildinfo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai_common/src/xai_errstr.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai_common/src/cnn_cast.c
)

set(LIBXAI_COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai_common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai_common/src
)

add_library(xai_common STATIC ${LIBXAI_COMMON_SOURCES})
target_include_directories(xai_common PUBLIC ${LIBXAI_COMMON_INCLUDE_DIRS})

# ============================================================================
# libxai - CNN kernels library
# ============================================================================
set(LIBXAI_SOURCES
    # Main convolution dispatcher (contains xaiConvolved3D)
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_conv.c
    # Convolution dispatcher and variants
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_conv_VQ.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_conv_MOD.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_conv_MOW.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_conv_SO.c
    # Dilated convolution VQ variants
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_VQ_MOW.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_VQ_MOD.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_VQ_MOD_S16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_VQ_MOW_S16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_VQ_SO.c
    # Dilated convolution non-VQ variants
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_MOW.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_MOD.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_MOD_S16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_MOW_S16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_dilated_conv_SO.c
    # Data transform and helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_datatransform.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src/cnn_helper.c
)

set(LIBXAI_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libxai/cnn/src
)

add_library(xai STATIC ${LIBXAI_SOURCES})
target_include_directories(xai PUBLIC ${LIBXAI_INCLUDE_DIRS})
target_include_directories(xai PRIVATE ${LIBXAI_COMMON_INCLUDE_DIRS})
target_link_libraries(xai PUBLIC xai_common)

# ============================================================================
# Export variables for parent CMakeLists.txt
# ============================================================================
set(XAI_INCLUDE_DIRS
    ${LIBXAI_INCLUDE_DIRS}
    ${LIBXAI_COMMON_INCLUDE_DIRS}
    CACHE INTERNAL "XAI include directories"
)

set(XAI_LIBRARIES xai xai_common CACHE INTERNAL "XAI libraries")
