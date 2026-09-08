// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Ethos-U NPU bring-up for the Corstone-320 (SSE-320) variant of the all-ops
// runner. Only compiled when the consumer project selects the Ethos-U backend
// (gen_cproject.py --ethos-u defines ALL_OPS_ETHOS_U); the Corstone-315 build
// has no NPU and links a no-op ethos_setup().
//
// Registers and initialises the Ethos-U driver so the ExecuTorch Ethos-U
// delegate can dispatch Vela command streams. Mirrors the reference
// board/Corstone-320/ethos_setup.c from cmsis-executorch-simple, adapted to a
// single fixed U85 instance.

#include <cstdio>

#if defined(ALL_OPS_ETHOS_U)

#include "RTE_Components.h"
#include CMSIS_device_header

#include "ethosu_driver.h"

// SSE-320 NPU0 hardware constants. These live in the BSP's Board/Platform
// headers (platform_base_address.h / platform_irq.h), which are only on the
// include path when the full board component is selected. This harness uses the
// bare device (no board layer), so define them directly -- they are fixed for
// the Corstone-320 subsystem.
#ifndef NPU0_APB_BASE_S
#define NPU0_APB_BASE_S 0x50004000UL
#endif
#ifndef NPU0_IRQn
#define NPU0_IRQn ((IRQn_Type)16)
#endif

// Ethos-U85 scratch/cache buffer. 384 KiB matches the reference and the
// Ethos_U85_SYS_DRAM_Mid / Shared_Sram Vela config used at export time.
#ifndef ETHOS_CACHE_BUF_SIZE
#define ETHOS_CACHE_BUF_SIZE 393216
#endif
#ifndef ETHOS_CACHE_BUF_ALIGNMENT
#define ETHOS_CACHE_BUF_ALIGNMENT 32
#endif

#define ETHOS_SECURE_ENABLE 1
#define ETHOS_PRIVILEGE_ENABLE 1

static struct ethosu_driver EthosDriver;
static uint8_t ethos_cache[ETHOS_CACHE_BUF_SIZE] __attribute__((
    section("ethos_cache_buf"),
    aligned(ETHOS_CACHE_BUF_ALIGNMENT)));

extern "C" void NPU0_Handler(void) {
  ethosu_irq_handler(&EthosDriver);
}

extern "C" void ethos_setup(void) {
  void* const ethos_base_addr = reinterpret_cast<void*>(NPU0_APB_BASE_S);
  int rval = ethosu_init(
      &EthosDriver,
      ethos_base_addr,
      ethos_cache,
      sizeof(ethos_cache),
      ETHOS_SECURE_ENABLE,
      ETHOS_PRIVILEGE_ENABLE);
  if (rval != 0) {
    printf("Failed to initialize Arm Ethos-U driver (rval=%d)\n", rval);
    return;
  }
  NVIC_EnableIRQ(NPU0_IRQn);

  struct ethosu_hw_info hw_info;
  ethosu_get_hw_info(&EthosDriver, &hw_info);
  printf(
      "Ethos-U ready: arch v%u.%u.%u, MACs/cc %u\n",
      hw_info.version.arch_major_rev,
      hw_info.version.arch_minor_rev,
      hw_info.version.arch_patch_rev,
      static_cast<unsigned>(1u << hw_info.cfg.macs_per_cc));
}

#else // !ALL_OPS_ETHOS_U -- Corstone-315 (no NPU)

extern "C" void ethos_setup(void) {}

#endif
