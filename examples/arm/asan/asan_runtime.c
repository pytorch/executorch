/* Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * A lightweight AddressSanitizer runtime tailored for the ExecuTorch bare
 * metal examples. The goal is to provide basic memory safety diagnostics while
 * keeping the runtime self-contained.
 *
 * This implementation shares the following characteristics:
 *   * Shadow memory resolution is 16 bytes per shadow byte.
 *   * Only coarse grained poisoning is implemented. Consumers should rely on
 *     __asan_poison_memory_region / __asan_unpoison_memory_region to describe
 *     invalid regions (for example heap red-zones).
 *   * Stack poisoning is not implemented: the stack malloc/free stubs fall back
 *     to the compiler inserted slow path. This keeps the runtime small while
 *     still enabling heap / global diagnostics.
 *   * The runtime prints diagnostics and traps on the first detected error.
 *
 * Note that this does not aim to be a drop-in replacement for compiler-rt's
 * runtime. It is intentionally minimal to suit resource constrained bare-metal
 * targets and to mirror the structure of the existing ubsan runtime.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef ASAN_RUNTIME_PREFIX
#define ASAN_RUNTIME_PREFIX "[ASAN] "
#endif

/* Stringification needs two layers so macro arguments expand before '#'. */
#define ASAN_STRINGIZE_IMPL(x) #x
#define ASAN_STRINGIZE(x) ASAN_STRINGIZE_IMPL(x)

/* Memory map extracted from the Corstone linker scripts. Update with care. */
#define ASAN_ITCM_START 0x10000000u
#define ASAN_ITCM_SIZE 0x00080000u

#define ASAN_BROM_START 0x11000000u
#define ASAN_BROM_SIZE 0x00020000u

#define ASAN_BRAM_START 0x12000000u
#define ASAN_BRAM_SIZE 0x00200000u

#define ASAN_DTCM_START 0x30000000u
#define ASAN_DTCM_SIZE 0x00080000u

#define ASAN_SRAM_START 0x31000000u
#define ASAN_SRAM_SIZE 0x00200000u

#define ASAN_DDR_START 0x70000000u
#define ASAN_DDR_SIZE 0x10000000u

/* Shadow setup: 16 bytes of application memory are represented by 1 byte. */
#define ASAN_SHADOW_SCALE 4u
#define ASAN_SHADOW_GRANULARITY (1u << ASAN_SHADOW_SCALE)
#define ASAN_SHADOW_MASK (ASAN_SHADOW_GRANULARITY - 1u)

#define ASAN_SHADOW_SIZE(region_size) (((region_size) + ASAN_SHADOW_MASK) >> ASAN_SHADOW_SCALE)

#define ASAN_SHADOW_SIZE_ITCM ASAN_SHADOW_SIZE(ASAN_ITCM_SIZE)
#define ASAN_SHADOW_SIZE_BROM ASAN_SHADOW_SIZE(ASAN_BROM_SIZE)
#define ASAN_SHADOW_SIZE_BRAM ASAN_SHADOW_SIZE(ASAN_BRAM_SIZE)
#define ASAN_SHADOW_SIZE_DTCM ASAN_SHADOW_SIZE(ASAN_DTCM_SIZE)
#define ASAN_SHADOW_SIZE_SRAM ASAN_SHADOW_SIZE(ASAN_SRAM_SIZE)
#define ASAN_SHADOW_SIZE_DDR ASAN_SHADOW_SIZE(ASAN_DDR_SIZE)

#define ASAN_SHADOW_TOTAL_SIZE \
  (ASAN_SHADOW_SIZE_ITCM + ASAN_SHADOW_SIZE_BROM + ASAN_SHADOW_SIZE_BRAM + \
   ASAN_SHADOW_SIZE_DTCM + ASAN_SHADOW_SIZE_SRAM + ASAN_SHADOW_SIZE_DDR)

/* Shadow memory lives in .asan_shadow so the linker can park it in DDR. */
__attribute__((section(".asan_shadow"), aligned(16)))
static uint8_t g_asan_shadow[ASAN_SHADOW_TOTAL_SIZE];

typedef struct {
  uintptr_t start;
  uintptr_t end;
  uint8_t* shadow;
  size_t shadow_size;
  const char* name;
} asan_region_t;

static asan_region_t g_regions[] = {
    {ASAN_ITCM_START,
     ASAN_ITCM_START + ASAN_ITCM_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_ITCM,
     "ITCM"},
    {ASAN_BROM_START,
     ASAN_BROM_START + ASAN_BROM_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_BROM,
     "BROM"},
    {ASAN_BRAM_START,
     ASAN_BRAM_START + ASAN_BRAM_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_BRAM,
     "BRAM"},
    {ASAN_DTCM_START,
     ASAN_DTCM_START + ASAN_DTCM_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_DTCM,
     "DTCM"},
    {ASAN_SRAM_START,
     ASAN_SRAM_START + ASAN_SRAM_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_SRAM,
     "SRAM"},
    {ASAN_DDR_START,
     ASAN_DDR_START + ASAN_DDR_SIZE,
     NULL,
     ASAN_SHADOW_SIZE_DDR,
     "DDR"},
};

static bool g_asan_initialized = false;

typedef struct {
  uintptr_t start;
  uintptr_t end;
  const asan_region_t* region;
} asan_check_result;

typedef struct {
  uintptr_t start;
  uintptr_t end;
  const char* name;
} asan_peripheral_range_t;

/* Allow-list for memory-mapped peripherals that must bypass checking. */
static const asan_peripheral_range_t g_peripheral_ranges[] = {
    {0xE0000000u, 0xE0100000u, "SCS"},
    {0x40000000u, 0x60000000u, "Peripheral"},
};

static bool asan_region_contains(const asan_region_t* region,
                                 uintptr_t begin,
                                 uintptr_t end) {
  return region && begin >= region->start && end <= region->end;
}

static const char* asan_region_name(const asan_region_t* region) {
  return region ? region->name : "<unknown>";
}

static void asan_shadow_set(asan_region_t* region,
                            uintptr_t begin,
                            size_t size,
                            uint8_t value) {
  if (!region || size == 0 || region->shadow == NULL) {
    return;
  }
  uintptr_t offset = begin - region->start;
  size_t shadow_begin = offset >> ASAN_SHADOW_SCALE;
  size_t shadow_end =
      (offset + size + ASAN_SHADOW_MASK) >> ASAN_SHADOW_SCALE;
  if (shadow_end > region->shadow_size) {
    shadow_end = region->shadow_size;
  }
  if (shadow_begin >= shadow_end) {
    return;
  }
  memset(region->shadow + shadow_begin, value, shadow_end - shadow_begin);
}

static bool asan_shadow_is_poisoned(const asan_region_t* region,
                                    uintptr_t begin,
                                    size_t size) {
  if (!region || size == 0 || region->shadow == NULL) {
    return true;
  }
  uintptr_t offset = begin - region->start;
  size_t shadow_begin = offset >> ASAN_SHADOW_SCALE;
  size_t shadow_end =
      (offset + size + ASAN_SHADOW_MASK) >> ASAN_SHADOW_SCALE;
  if (shadow_end > region->shadow_size) {
    shadow_end = region->shadow_size;
  }
  for (size_t idx = shadow_begin; idx < shadow_end; ++idx) {
    if (region->shadow[idx] != 0) {
      return true;
    }
  }
  return false;
}

static asan_region_t* asan_find_region(uintptr_t begin, uintptr_t end) {
  for (size_t i = 0; i < sizeof(g_regions) / sizeof(g_regions[0]); ++i) {
    asan_region_t* region = &g_regions[i];
    if (asan_region_contains(region, begin, end)) {
      return region;
    }
  }
  return NULL;
}

static void asan_report_error(const char* kind,
                              void* addr,
                              size_t size,
                              const char* reason,
                              const asan_region_t* region) {
  printf(ASAN_RUNTIME_PREFIX "%s of size %zu at %p failed: %s (region=%s)\n",
         kind,
         size,
         addr,
         reason,
         asan_region_name(region));
  fflush(stdout);
#if defined(__GNUC__)
  __builtin_trap();
#else
  while (1) {
  }
#endif
}

static bool asan_check_address(const char* kind, void* addr, size_t size) {
  if (!g_asan_initialized) {
    return true;
  }
  uintptr_t begin = (uintptr_t)addr;
  uintptr_t end = begin + size;
  for (size_t i = 0; i < sizeof(g_peripheral_ranges) / sizeof(g_peripheral_ranges[0]); ++i) {
    const asan_peripheral_range_t* range = &g_peripheral_ranges[i];
    if (begin >= range->start && end <= range->end) {
      return true;
    }
  }
  if (end < begin) {
    asan_report_error(kind, addr, size, "overflow in address range", NULL);
    return false;
  }
  asan_region_t* region = asan_find_region(begin, end);
  if (!region) {
    asan_report_error(kind, addr, size, "address outside tracked regions", NULL);
    return false;
  }
  if (asan_shadow_is_poisoned(region, begin, size)) {
    asan_report_error(kind, addr, size, "poisoned shadow", region);
    return false;
  }
  return true;
}

/* ----------- Sanitizer runtime entry points ----------- */

int __asan_option_detect_stack_use_after_return = 0;

void __asan_init(void) {
  if (g_asan_initialized) {
    return;
  }
  uint8_t* shadow_cursor = g_asan_shadow;
  for (size_t i = 0; i < sizeof(g_regions) / sizeof(g_regions[0]); ++i) {
    g_regions[i].shadow = shadow_cursor;
    shadow_cursor += g_regions[i].shadow_size;
    /* Mark entire region as accessible by default. */
    asan_shadow_set(&g_regions[i], g_regions[i].start, g_regions[i].end - g_regions[i].start, 0);
  }
  g_asan_initialized = true;
}

void __asan_version_mismatch_check_v8(void) {}

void __asan_handle_no_return(void) {}

void __asan_poison_memory_region(void* addr, size_t size) {
  if (!g_asan_initialized) {
    return;
  }
  uintptr_t begin = (uintptr_t)addr;
  uintptr_t end = begin + size;
  asan_region_t* region = asan_find_region(begin, end);
  if (!region) {
    return;
  }
  asan_shadow_set(region, begin, size, 0xFF);
}

void __asan_unpoison_memory_region(void* addr, size_t size) {
  if (!g_asan_initialized) {
    return;
  }
  uintptr_t begin = (uintptr_t)addr;
  uintptr_t end = begin + size;
  asan_region_t* region = asan_find_region(begin, end);
  if (!region) {
    return;
  }
  asan_shadow_set(region, begin, size, 0x00);
}

void __asan_alloca_poison(void* addr, size_t size) {
  if (!g_asan_initialized) {
    return;
  }
  __asan_poison_memory_region(addr, size);
}

void __asan_allocas_unpoison(void* top, void* bottom) {
  if (!g_asan_initialized) {
    return;
  }
  uintptr_t begin = (uintptr_t)bottom;
  uintptr_t end = (uintptr_t)top;
  if (end <= begin) {
    return;
  }
  __asan_unpoison_memory_region(bottom, end - begin);
}

#define ASAN_DEFINE_LOAD_NOABORT(N)                               \
  void __asan_load##N##_noabort(void* addr) {                     \
    (void)asan_check_address("load" ASAN_STRINGIZE(N), addr, N); \
  }

ASAN_DEFINE_LOAD_NOABORT(1)
ASAN_DEFINE_LOAD_NOABORT(2)
ASAN_DEFINE_LOAD_NOABORT(4)
ASAN_DEFINE_LOAD_NOABORT(8)
ASAN_DEFINE_LOAD_NOABORT(16)

#undef ASAN_DEFINE_LOAD_NOABORT

void __asan_loadN_noabort(void* addr, size_t size) {
  (void)asan_check_address("loadN", addr, size);
}

#define ASAN_DEFINE_STORE_NOABORT(N)                                \
  void __asan_store##N##_noabort(void* addr) {                      \
    (void)asan_check_address("store" ASAN_STRINGIZE(N), addr, N); \
  }

ASAN_DEFINE_STORE_NOABORT(1)
ASAN_DEFINE_STORE_NOABORT(2)
ASAN_DEFINE_STORE_NOABORT(4)
ASAN_DEFINE_STORE_NOABORT(8)
ASAN_DEFINE_STORE_NOABORT(16)

#undef ASAN_DEFINE_STORE_NOABORT

void __asan_storeN_noabort(void* addr, size_t size) {
  (void)asan_check_address("storeN", addr, size);
}

/* The compiler still emits the reporting entry points. Delegate to the same helper. */
#define ASAN_DEFINE_REPORT_LOAD(N)                                            \
  void __asan_report_load##N(void* addr) {                                    \
    asan_report_error("load" ASAN_STRINGIZE(N), addr, N, "reported hook", NULL); \
  }

ASAN_DEFINE_REPORT_LOAD(1)
ASAN_DEFINE_REPORT_LOAD(2)
ASAN_DEFINE_REPORT_LOAD(4)
ASAN_DEFINE_REPORT_LOAD(8)
ASAN_DEFINE_REPORT_LOAD(16)

#undef ASAN_DEFINE_REPORT_LOAD

void __asan_report_load_n(void* addr, size_t size) {
  asan_report_error("loadN", addr, size, "reported hook", NULL);
}

#define ASAN_DEFINE_REPORT_STORE(N)                                             \
  void __asan_report_store##N(void* addr) {                                     \
    asan_report_error("store" ASAN_STRINGIZE(N), addr, N, "reported hook", NULL); \
  }

ASAN_DEFINE_REPORT_STORE(1)
ASAN_DEFINE_REPORT_STORE(2)
ASAN_DEFINE_REPORT_STORE(4)
ASAN_DEFINE_REPORT_STORE(8)
ASAN_DEFINE_REPORT_STORE(16)

#undef ASAN_DEFINE_REPORT_STORE

void __asan_report_store_n(void* addr, size_t size) {
  asan_report_error("storeN", addr, size, "reported hook", NULL);
}

/* Stubbed APIs required by the instrumentation.
 * Intentional no-ops: we rely on the compiler slow path to handle stack
 * poisoning so the runtime stays minimal. */
#define ASAN_STACK_MALLOC_FREE(N)                                        \
  void* __asan_stack_malloc_##N(size_t size) {                           \
    (void)size;                                                          \
    return NULL; /* fall back to compiler slow path */                   \
  }                                                                      \
                                                                         \
  void __asan_stack_free_##N(void* ptr, size_t size) {                   \
    (void)ptr;                                                           \
    (void)size;                                                          \
  }

ASAN_STACK_MALLOC_FREE(0)
ASAN_STACK_MALLOC_FREE(1)
ASAN_STACK_MALLOC_FREE(2)
ASAN_STACK_MALLOC_FREE(3)
ASAN_STACK_MALLOC_FREE(4)
ASAN_STACK_MALLOC_FREE(5)
ASAN_STACK_MALLOC_FREE(6)
ASAN_STACK_MALLOC_FREE(7)
ASAN_STACK_MALLOC_FREE(8)
ASAN_STACK_MALLOC_FREE(9)
ASAN_STACK_MALLOC_FREE(10)

#undef ASAN_STACK_MALLOC_FREE

struct __asan_global {
  uintptr_t beg;
  size_t size;
  size_t size_with_redzone;
  const char* name;
  const char* module_name;
  uintptr_t has_dynamic_init;
  uintptr_t location;
  uintptr_t odr_indicator;
};

void __asan_register_globals(struct __asan_global* globals, size_t n) {
  if (!g_asan_initialized) {
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    __asan_unpoison_memory_region((void*)globals[i].beg, globals[i].size);
  }
}

void __asan_unregister_globals(struct __asan_global* globals, size_t n) {
  if (!g_asan_initialized) {
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    __asan_poison_memory_region((void*)globals[i].beg, globals[i].size);
  }
}

void __asan_register_image_globals(struct __asan_global* globals,
                                   size_t n) {
  __asan_register_globals(globals, n);
}

void __asan_unregister_image_globals(struct __asan_global* globals,
                                     size_t n) {
  __asan_unregister_globals(globals, n);
}

/* Weak aliases so that missing hooks do not cause link failures. */
void __asan_before_dynamic_init(const char* module_name) {
  (void)module_name;
}

void __asan_after_dynamic_init(void) {}

__attribute__((constructor)) static void asan_runtime_constructor(void) {
  __asan_init();
}
