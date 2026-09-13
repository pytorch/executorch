# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXECUTABLE_LIBRARY_PREFIX)
set(CMAKE_EXECUTABLE_LIBRARY_SUFFIX)

# CURRENT DIRECTORY
set(ProjDirPath ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# Skip link step during compiler check (bare-metal cross-compilation).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(executorch_cifarnet)

enable_language(ASM)

set(MCUX_SDK_PROJECT_NAME executorch_cifarnet.elf)

set(EXECUTORCH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

# CPU and FPU flags required for Cortex-M33 with single-precision FPU.
set(CPU_FLAGS "-mcpu=cortex-m33 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16")
set(CPU_DEFINES
    "-DCPU_MIMXRT798SGFOA_cm33_core0 -DCPU_MIMXRT798SGFOB_cm33_core0 \
     -DMIMXRT798S_cm33_core0_SERIES -DMCUXPRESSO_SDK \
     -D__STARTUP_INITIALIZE_NONCACHEDATA -D__STARTUP_CLEAR_BSS \
     -DDSP_IMAGE_COPY_TO_RAM=1 -DBOOT_HEADER_ENABLE=1 \
     -DEIQ_EXAMPLE_HSRUN_CLOCK -DMCUX_META_BUILD \
     -DPRINTF_ADVANCED_ENABLE=1 -DPRINTF_FLOAT_ENABLE=1 -DNO_HEAP_USAGE=1 \
     -DSDK_DEBUGCONSOLE=1 -DSDK_I2C_BASED_COMPONENT_USED=1"
)
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS}   ${CPU_FLAGS} ${CPU_DEFINES} -fno-common -ffunction-sections -fdata-sections -fno-builtin -mapcs -std=gnu99"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${CPU_FLAGS} ${CPU_DEFINES} -fno-common -ffunction-sections -fdata-sections -fno-builtin -mapcs -fno-rtti -fno-exceptions"
)
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${CPU_FLAGS} ${CPU_DEFINES}")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${CPU_FLAGS} -fno-common -ffunction-sections -fdata-sections -fno-builtin -mapcs -Wl,--gc-sections -Wl,-static -specs=nano.specs -specs=nosys.specs -T\"${SdkRootDirPath}/examples/_boards/mimxrt700evk/eiq_examples/executorch_cifarnet/cm33_core0/gcc/MIMXRT798Sxxxx_cm33_core0_flash.ld\" -static"
)

add_executable(
  ${MCUX_SDK_PROJECT_NAME}
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/flash_config/flash_config.c
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/eiq_examples/executorch_cifarnet/cm33_core0/hardware_init.c
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/eiq_examples/executorch_cifarnet/cm33_core0/pin_mux.c
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/board.c
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/pmic_support.c
  ${SdkRootDirPath}/examples/_boards/mimxrt700evk/common/clock/cm33_core0/clock_config.c
  ${SdkRootDirPath}/examples/eiq_examples/executorch_cifarnet/main.cpp
  ${SdkRootDirPath}/examples/eiq_examples/executorch_cifarnet/RegisterKernels.cpp
  ${SdkRootDirPath}/examples/eiq_examples/common/timer.c
  ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/startup_MIMXRT798S_cm33_core0.c
  ${EXECUTORCH_ROOT_DIR}/backends/nxp/runtime/NeutronBackend.cpp
  ${SdkRootDirPath}/middleware/tfm/tf-m/platform/ext/common/syscalls_stub.c
  # Device-level drivers (clock, power, reset, system init).
  ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/drivers/fsl_clock.c
  ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/drivers/fsl_power.c
  ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/drivers/fsl_reset.c
  ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/system_MIMXRT798S_cm33_core0.c
  # Common ARM driver (provides SDK_DelayAtLeastUs).
  ${SdkRootDirPath}/drivers/common/fsl_common_arm.c
  # Peripheral drivers.
  ${SdkRootDirPath}/drivers/cache/xcache/fsl_cache.c
  ${SdkRootDirPath}/drivers/lpflexcomm/fsl_lpflexcomm.c
  ${SdkRootDirPath}/drivers/lpflexcomm/lpi2c/fsl_lpi2c.c
  ${SdkRootDirPath}/drivers/lpflexcomm/lpuart/fsl_lpuart.c
  ${SdkRootDirPath}/drivers/gpio/fsl_gpio.c
  ${SdkRootDirPath}/drivers/glikey/fsl_glikey.c
  # UART HAL adapter (provides HAL_UartInit etc.).
  ${SdkRootDirPath}/components/uart/fsl_adapter_lpuart.c
  # PMIC driver.
  ${SdkRootDirPath}/components/pmic/pca9422/fsl_pca9422.c
  # Debug console (provides DbgConsole_Init/Printf).
  ${SdkRootDirPath}/components/debug_console_lite/fsl_debug_console.c
)

target_include_directories(
  ${MCUX_SDK_PROJECT_NAME}
  PRIVATE
    ${MODEL_DIR}
    ${SdkRootDirPath}/arch/arm/CMSIS/Core/Include
    ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S
    ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/cm33_core0
    ${SdkRootDirPath}/devices/RT/RT700/MIMXRT798S/drivers
    ${SdkRootDirPath}/devices/RT/RT700/periph
    ${SdkRootDirPath}/drivers/common
    ${SdkRootDirPath}/components/pmic/pca9422
    ${SdkRootDirPath}/components/uart
    ${SdkRootDirPath}/components/debug_console_lite
    ${SdkRootDirPath}/examples/_boards/mimxrt700evk
    ${SdkRootDirPath}/examples/_boards/mimxrt700evk/flash_config
    ${SdkRootDirPath}/examples/_boards/mimxrt700evk/eiq_examples/executorch_cifarnet/cm33_core0
    ${SdkRootDirPath}/examples/_boards/mimxrt700evk/common/clock/cm33_core0
    ${SdkRootDirPath}/examples/eiq_examples/executorch_cifarnet
    ${SdkRootDirPath}/examples/eiq_examples/common
    ${SdkRootDirPath}/examples/_boards/mimxrt700evk/eiq_examples/executorch_cifarnet/npu
    ${SdkRootDirPath}/drivers/cache/xcache
    ${SdkRootDirPath}/drivers/gpio
    ${SdkRootDirPath}/drivers/lpflexcomm
    ${SdkRootDirPath}/drivers/lpflexcomm/lpuart
    ${SdkRootDirPath}/drivers/lpflexcomm/lpi2c
    ${SdkRootDirPath}/drivers/xspi
    ${SdkRootDirPath}/drivers/reset
    ${SdkRootDirPath}/drivers/clock
    ${SdkRootDirPath}/drivers/glikey
    ${SdkRootDirPath}/drivers/mu1
    ${SdkRootDirPath}/drivers/power
    ${SdkRootDirPath}/drivers/iopctl
    ${SdkRootDirPath}/components/str
)

set(EXECUTORCH_BUILD_PYBIND OFF)
set(EXECUTORCH_BUILD_TESTS OFF)
set(EXECUTORCH_BUILD_DEVTOOLS OFF)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_CPUINFO OFF)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set(EXECUTORCH_BUILD_PORTABLE_OPS ON)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
add_subdirectory(${EXECUTORCH_ROOT_DIR} EXCLUDE_FROM_ALL executorch)

target_link_libraries(
  ${MCUX_SDK_PROJECT_NAME}
  PRIVATE -Wl,--start-group
          executorch
          executorch_core
          extension_runner_util
          quantized_kernels
          portable_kernels
          ${NEUTRON_LIB_DIR}/libNeutronDriver.a
          ${NEUTRON_LIB_DIR}/libNeutronFirmware.a
          -Wl,--end-group
)
