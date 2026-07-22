# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# config to select component, the format is CONFIG_USE_${component} Please refer
# to cmake files below to get available components:
# ${SdkRootDirPath}/devices/MIMXRT798S/all_lib_device.cmake

set(CONFIG_COMPILER gcc)
set(CONFIG_TOOLCHAIN armgcc)
set(CONFIG_USE_COMPONENT_CONFIGURATION false)
set(CONFIG_USE_driver_flash_config true)
set(CONFIG_USE_CMSIS_Include_core_cm true)
set(CONFIG_USE_device_CMSIS true)
set(CONFIG_USE_device_system true)
set(CONFIG_USE_device_startup true)
set(CONFIG_USE_driver_clock true)
set(CONFIG_USE_driver_dsp true)
set(CONFIG_USE_driver_iopctl_soc true)
set(CONFIG_USE_driver_power true)
set(CONFIG_USE_driver_reset true)
set(CONFIG_USE_driver_cache_xcache true)
set(CONFIG_USE_driver_common true)
set(CONFIG_USE_driver_glikey true)
set(CONFIG_USE_driver_gpio true)
set(CONFIG_USE_driver_lpflexcomm true)
set(CONFIG_USE_driver_lpflexcomm_lpuart true)
set(CONFIG_USE_driver_lpflexcomm_lpi2c true)
set(CONFIG_USE_driver_xspi true)
set(CONFIG_USE_utility_assert_lite true)
set(CONFIG_USE_utilities_misc_utilities true)
set(CONFIG_USE_utility_str true)
set(CONFIG_USE_utility_debug_console_lite true)
set(CONFIG_USE_component_lpuart_adapter true)
set(CONFIG_USE_driver_pca9422 true)
set(CONFIG_CORE cm33)
set(CONFIG_DEVICE MIMXRT798S)
set(CONFIG_BOARD mimxrt700evk)
set(CONFIG_KIT mimxrt700evk)
set(CONFIG_DEVICE_ID MIMXRT798S)
set(CONFIG_FPU SP_FPU)
set(CONFIG_DSP DSP)
set(CONFIG_CORE_ID cm33_core0)
set(CONFIG_TRUSTZONE TZ)
set(CONFIG_USE_driver_mu1 true)
