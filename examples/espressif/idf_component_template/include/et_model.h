/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * et_model: a small C wrapper over the ExecuTorch runtime (Program/Method)
 * for ESP-IDF applications that just want to run a .pte from flash.
 *
 * This header carries no ESP-IDF or C++ types. The model bytes are used
 * in place (typically flash XIP) and must be 16-byte aligned, every
 * method arena is allocated with heap_caps in et_model_load_method, and
 * the inference path (set_input/execute/output) does not allocate
 * unless a kernel requests scratch memory and temp_arena_size is 0.
 *
 *   et_model_config_t cfg = ET_MODEL_CONFIG_DEFAULT(model_pte,
 *                                                   model_pte_size);
 *   cfg.planned_location = ET_MEM_INTERNAL;   // always-on arenas in SRAM
 *   et_model_t *m = et_model_create(&cfg);
 *   if (!m || !et_model_load_method(m, NULL)) { handle error }
 *   et_model_set_input(m, 0, frame, sizeof(frame));
 *   et_model_execute(m);
 *   size_t n; const void *out = et_model_output_data(m, 0, &n);
 *
 * Method state (mutable buffers, e.g. streaming-model history) persists
 * across et_model_execute calls on the same loaded method, as in the C++
 * API. Loading a method again resets that state.
 *
 * Not thread-safe: use one et_model_t from one task, like the underlying
 * executorch::runtime::Method.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Where an arena is allocated (mapped to heap_caps capabilities). */
typedef enum {
  ET_MEM_DEFAULT = 0, /* MALLOC_CAP_DEFAULT: IDF heap policy decides */
  ET_MEM_INTERNAL = 1, /* internal SRAM: fastest, scarcest */
  ET_MEM_PSRAM = 2, /* external PSRAM: fails if the board has none */
} et_mem_location_t;

typedef struct {
  /* .pte bytes, 16-byte aligned; must outlive the et_model_t. */
  const uint8_t* pte;
  size_t pte_size;
  /* Method allocator pool in bytes (method bookkeeping and non-planned
   * tensors); 0 selects the default (256 KiB). Planned activation buffers
   * and kernel scratch are allocated separately. */
  size_t arena_size;
  et_mem_location_t arena_location;
  /* Scratch pool for kernels and delegates that request temporary memory
   * (e.g. FFT, topk), reset after every kernel. 0 (default) allocates no
   * pool; the runtime then mallocs scratch through the PAL during
   * execute, only for kernels that need it. */
  size_t temp_arena_size;
  et_mem_location_t temp_location;
  /* Memory-planned activation/state buffers. Always-on models usually
   * want ET_MEM_INTERNAL; large models may need ET_MEM_PSRAM. */
  et_mem_location_t planned_location;
  /* Fully verify the program flatbuffer at load. Costs load time; use it
   * for .pte files that are not trusted at build time (e.g. OTA). */
  bool verify;
} et_model_config_t;

#define ET_MODEL_CONFIG_DEFAULT(pte_ptr, pte_len)                        \
  {                                                                      \
    .pte = (pte_ptr), .pte_size = (pte_len), .arena_size = 0,            \
    .arena_location = ET_MEM_DEFAULT, .temp_arena_size = 0,              \
    .temp_location = ET_MEM_DEFAULT, .planned_location = ET_MEM_DEFAULT, \
    .verify = false,                                                     \
  }

typedef struct et_model et_model_t;

/* Loads the program (header check only unless cfg->verify is set).
 * Returns NULL on failure (reason is logged via ESP_LOG). Does not load
 * a method yet. */
et_model_t* et_model_create(const et_model_config_t* cfg);
void et_model_destroy(et_model_t* model);

size_t et_model_num_methods(et_model_t* model);
/* Name of method `index`, owned by the program; NULL if out of range. */
const char* et_model_method_name(et_model_t* model, size_t index);

/* Loads `name` (NULL = first method), allocating its arenas and planned
 * buffers. Replaces any previously loaded method and resets mutable
 * state; on failure no method is loaded. Methods with more than 16
 * memory-planned buffers are rejected. */
bool et_model_load_method(et_model_t* model, const char* name);

/* Planned-buffer footprint of the named (NULL = loaded) method, for
 * sizing sdkconfig/memory budgets without loading it. Returns false if
 * the method does not exist, or if `name` is NULL and no method is
 * loaded. Either out pointer may be NULL. */
bool et_model_planned_info(
    et_model_t* model,
    const char* name,
    size_t* out_num_buffers,
    size_t* out_total_bytes);

size_t et_model_num_inputs(et_model_t* model);
size_t et_model_num_outputs(et_model_t* model);

/* Tensor metadata, mirroring ExecuTorch's TensorInfo. scalar_type uses
 * the executorch::aten::ScalarType numbering (see
 * executorch/runtime/core/portable_type/scalar_type.h). */
#define ET_MODEL_MAX_DIMS 8
typedef struct {
  int32_t scalar_type;
  size_t ndim;
  int32_t dims[ET_MODEL_MAX_DIMS];
  size_t nbytes;
} et_tensor_info_t;

/* Fail for non-tensor values and tensors with more than
 * ET_MODEL_MAX_DIMS dims. */
bool et_model_input_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out);
bool et_model_output_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out);

/* Sets tensor input `index` from `data`. `nbytes` must equal the input's
 * exported nbytes (et_model_input_info); dynamic shapes are not supported.
 * Memory-planned inputs (the export default) are copied, so `data` may be
 * reused immediately or live in flash. Inputs exported without memory
 * planning keep pointing at `data`, which must then stay valid and
 * unmodified until the input is set again. Non-tensor inputs cannot be
 * set and keep their exported values. */
bool et_model_set_input(
    et_model_t* model,
    size_t index,
    const void* data,
    size_t nbytes);

bool et_model_execute(et_model_t* model);

/* Output tensor storage after a successful execute; valid until the next
 * execute/load/destroy. Returns NULL on error. *out_nbytes may be NULL. */
const void*
et_model_output_data(et_model_t* model, size_t index, size_t* out_nbytes);

/* Human-readable reason for the most recent failed call on `model`;
 * never NULL. */
const char* et_model_last_error(et_model_t* model);

#ifdef __cplusplus
}
#endif
