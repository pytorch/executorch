/*
 * memory_manager.h
 *
 *  Created on: Nov 6, 2025
 *      Author: sraut
 *
 *  Description: Dynamic memory allocator for DRAM0, DRAM1, and local SRAM regions
 *               Provides simple arena-style allocation with 64-byte alignment
 */

#ifndef MEMORY_MANAGER_H_
#define MEMORY_MANAGER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "../../operators/layer_configs.h"  // For IDMA_BUFFER_SIZE_DRAM0/DRAM1

// ============================================================================
// Memory Configuration
// ============================================================================

// Cache-mode padded input buffer size (in system memory)
// Must fit the largest padded input tensor for cache-mode layers
// For ResNet layers that don't fit in DRAM tiling (e.g., 56x56x128)
#ifndef CACHE_PADDED_INPUT_SIZE
#define CACHE_PADDED_INPUT_SIZE  (1024 * 1024)  // 1 MB max
#endif

// ============================================================================
// Dynamic Memory Allocator for DRAM0 and DRAM1
// ============================================================================

// Memory pools placed in specific DRAM sections
// Declared extern here, defined in memory_manager.c
extern uint8_t dram0_pool[IDMA_BUFFER_SIZE_DRAM0];
extern uint8_t dram1_pool[IDMA_BUFFER_SIZE_DRAM1];

// Cache-mode padded input buffer (in system memory)
// Used by cache-mode kernels for edge padding
extern int8_t cache_padded_input[CACHE_PADDED_INPUT_SIZE];

/**
 * @brief Allocate DRAM buffer with SIMD alignment
 * @param size Size in bytes to allocate
 * @param dram_bank Which DRAM bank (0 or 1)
 * @param dram0_used Pointer to current dram0 usage counter
 * @param dram1_used Pointer to current dram1 usage counter
 * @return Pointer to allocated buffer
 */
int8_t* allocate_dram_buffer(int size, int dram_bank, int* dram0_used, int* dram1_used);

/**
 * @brief Get pointer to cache-mode padded input buffer
 * @return Pointer to the padded input buffer (aligned, in system memory)
 */
static inline int8_t* get_cache_padded_input(void) {
    return cache_padded_input;
}

/**
 * @brief Get size of cache-mode padded input buffer
 * @return Size in bytes
 */
static inline size_t get_cache_padded_input_size(void) {
    return CACHE_PADDED_INPUT_SIZE;
}

#endif /* MEMORY_MANAGER_H_ */
