cmake_minimum_required(VERSION 3.19)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(COMMON_INCLUDE_DIRS ${EXECUTORCH_ROOT}/..)

if(EXECUTORCH_BUILD_TENSORRT)

  find_package(CUDAToolkit REQUIRED)

  find_library(NVONNXPARSER_LIBRARY nvonnxparser
    HINTS ${TENSORRT_HOME}/lib ${TENSORRT_HOME}/lib64
          /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  )

  # TensorRT executor runner
  add_executable(
    tensorrt_executor_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_executor_runner.cpp
  )

  target_include_directories(
    tensorrt_executor_runner
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
           $<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>
  )

  target_link_libraries(
    tensorrt_executor_runner
    PRIVATE executorch
            extension_module_static
            extension_tensor
            extension_data_loader
            portable_kernels
  )

  target_link_options(
    tensorrt_executor_runner
    PRIVATE
      "SHELL:LINKER:--whole-archive $<TARGET_FILE:tensorrt_backend> LINKER:--no-whole-archive"
  )
  target_link_libraries(tensorrt_executor_runner PRIVATE CUDA::cudart)
  if(TENSORRT_LIBRARY)
    target_link_libraries(tensorrt_executor_runner PRIVATE ${TENSORRT_LIBRARY})
  endif()
  add_dependencies(tensorrt_executor_runner tensorrt_backend)

  target_compile_options(tensorrt_executor_runner PRIVATE -frtti -fexceptions)

  install(TARGETS tensorrt_executor_runner DESTINATION ${CMAKE_INSTALL_BINDIR})

  # TensorRT benchmark runner
  add_executable(benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)

  target_include_directories(
    benchmark
    PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
           $<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>
  )

  target_link_libraries(
    benchmark
    PRIVATE executorch
            extension_module_static
            extension_tensor
            portable_kernels
  )

  target_link_options(
    benchmark
    PRIVATE
      "SHELL:LINKER:--whole-archive $<TARGET_FILE:tensorrt_backend> LINKER:--no-whole-archive"
  )
  target_link_libraries(benchmark PRIVATE CUDA::cudart)
  if(TENSORRT_LIBRARY)
    target_link_libraries(benchmark PRIVATE ${TENSORRT_LIBRARY})
  endif()
  if(NVONNXPARSER_LIBRARY)
    target_link_libraries(benchmark PRIVATE ${NVONNXPARSER_LIBRARY})
  endif()
  add_dependencies(benchmark tensorrt_backend)

  target_compile_options(benchmark PRIVATE -frtti -fexceptions)

  install(TARGETS benchmark DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
