# cortex-m0plus cmake

if(NOT DEFINED EXECUTORCH_BUILD_ARM_BAREMETAL)
  # If not defined, assume we're building standalone
  set(EXECUTORCH_BUILD_ARM_BAREMETAL ON)
endif()
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR cortex-m0plus)

if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER arm-none-eabi-gcc CACHE STRING "C compiler")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER arm-none-eabi-g++ CACHE STRING "C++ compiler")
endif()

set(CPU_FLAGS "-mcpu=cortex-m0plus -mthumb -mfloat-abi=soft")
# C flags (no RTTI or exceptions here, since RTTI is C++-only)
set(CMAKE_C_FLAGS "${CPU_FLAGS} -O2 -ffunction-sections -fdata-sections -fno-exceptions -fno-unwind-tables")

# C++ flags (RTTI-related flags go here)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fno-rtti -fno-use-cxa-atexit -ffunction-sections -fdata-sections")

# Linker flags
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -nostartfiles -flto")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "-nostartfiles")
