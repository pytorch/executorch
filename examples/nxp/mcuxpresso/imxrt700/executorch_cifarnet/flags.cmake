# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT DEFINED FPU)
  set(FPU "-mfloat-abi=hard -mfpu=fpv5-sp-d16")
endif()

if(NOT DEFINED SPECS)
  set(SPECS "--specs=nano.specs --specs=nosys.specs")
endif()

if(NOT DEFINED DEBUG_CONSOLE_CONFIG)
  set(DEBUG_CONSOLE_CONFIG "-DSDK_DEBUGCONSOLE=1")
endif()

set(CMAKE_ASM_FLAGS_FLASH_DEBUG
    " \
    ${CMAKE_ASM_FLAGS_FLASH_DEBUG} \
    -D__STARTUP_INITIALIZE_NONCACHEDATA \
    -D__STARTUP_CLEAR_BSS \
    -DDSP_IMAGE_COPY_TO_RAM=1 \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -g \
    -mthumb \
    -mcpu=cortex-m33 \
    ${FPU} \
"
)
set(CMAKE_ASM_FLAGS_FLASH_RELEASE
    " \
    ${CMAKE_ASM_FLAGS_FLASH_RELEASE} \
    -D__STARTUP_INITIALIZE_NONCACHEDATA \
    -D__STARTUP_CLEAR_BSS \
    -DDSP_IMAGE_COPY_TO_RAM=1 \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -mthumb \
    -mcpu=cortex-m33 \
    ${FPU} \
"
)
set(CMAKE_C_FLAGS_FLASH_DEBUG
    " \
    ${CMAKE_C_FLAGS_FLASH_DEBUG} \
    -DDEBUG \
    -D__STARTUP_INITIALIZE_NONCACHEDATA \
    -D__STARTUP_CLEAR_BSS \
    -DDSP_IMAGE_COPY_TO_RAM=1 \
    -DEIQ_EXAMPLE_HSRUN_CLOCK \
    -DMCUX_META_BUILD \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -DBOOT_HEADER_ENABLE=1 \
    -DSDK_I2C_BASED_COMPONENT_USED=1 \
    -g \
    -O0 \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -std=gnu99 \
    -mcpu=cortex-m33 \
    -DPRINTF_ADVANCED_ENABLE=1 \
    -DPRINTF_FLOAT_ENABLE=1 \
    -DNO_HEAP_USAGE=1 \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
"
)
set(CMAKE_C_FLAGS_FLASH_RELEASE
    " \
    ${CMAKE_C_FLAGS_FLASH_RELEASE} \
    -DNDEBUG \
    -D__STARTUP_INITIALIZE_NONCACHEDATA \
    -D__STARTUP_CLEAR_BSS \
    -DDSP_IMAGE_COPY_TO_RAM=1 \
    -DEIQ_EXAMPLE_HSRUN_CLOCK \
    -DMCUX_META_BUILD \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -DBOOT_HEADER_ENABLE=1 \
    -DSDK_I2C_BASED_COMPONENT_USED=1 \
    -Os \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -std=gnu99 \
    -mcpu=cortex-m33 \
    -DPRINTF_ADVANCED_ENABLE=1 \
    -DPRINTF_FLOAT_ENABLE=1 \
    -DNO_HEAP_USAGE=1 \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
"
)
set(CMAKE_CXX_FLAGS_FLASH_DEBUG
    " \
    ${CMAKE_CXX_FLAGS_FLASH_DEBUG} \
    -DDEBUG \
    -DMCUX_META_BUILD \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -DBOOT_HEADER_ENABLE=1 \
    -g \
    -O0 \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -fno-rtti \
    -fno-exceptions \
    -mcpu=cortex-m33 \
    -DPRINTF_ADVANCED_ENABLE=1 \
    -DPRINTF_FLOAT_ENABLE=1 \
    -DNO_HEAP_USAGE=1 \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
"
)
set(CMAKE_CXX_FLAGS_FLASH_RELEASE
    " \
    ${CMAKE_CXX_FLAGS_FLASH_RELEASE} \
    -DNDEBUG \
    -DMCUX_META_BUILD \
    -DMCUXPRESSO_SDK \
    -DCPU_MIMXRT798SGFOA_cm33_core0 \
    -DCPU_MIMXRT798SGFOB_cm33_core0 \
    -DMIMXRT798S_cm33_core0_SERIES \
    -DBOOT_HEADER_ENABLE=1 \
    -Os \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -fno-rtti \
    -fno-exceptions \
    -mcpu=cortex-m33 \
    -DPRINTF_ADVANCED_ENABLE=1 \
    -DPRINTF_FLOAT_ENABLE=1 \
    -DNO_HEAP_USAGE=1 \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
"
)
set(CMAKE_EXE_LINKER_FLAGS_FLASH_DEBUG
    " \
    ${CMAKE_EXE_LINKER_FLAGS_FLASH_DEBUG} \
    -g \
    -Xlinker \
    -Map=output.map \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -Wl,--gc-sections \
    -Wl,-static \
    -Wl,--print-memory-usage \
    -mcpu=cortex-m33 \
    ${FPU} \
    ${SPECS} \
    -T\"${ProjDirPath}/MIMXRT798Sxxxx_cm33_core0_flash.ld\" -static \
"
)
set(CMAKE_EXE_LINKER_FLAGS_FLASH_RELEASE
    " \
    ${CMAKE_EXE_LINKER_FLAGS_FLASH_RELEASE} \
    -Xlinker \
    -Map=output.map \
    -Wall \
    -fno-common \
    -ffunction-sections \
    -fdata-sections \
    -fno-builtin \
    -mthumb \
    -mapcs \
    -Wl,--gc-sections \
    -Wl,-static \
    -Wl,--print-memory-usage \
    -mcpu=cortex-m33 \
    ${FPU} \
    ${SPECS} \
    -T\"${ProjDirPath}/MIMXRT798Sxxxx_cm33_core0_flash.ld\" -static \
"
)
