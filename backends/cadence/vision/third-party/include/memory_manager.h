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
#include "../../operators/conv/conv_layer_configs.h"  // For IDMA_BUFFER_SIZE_DRAM0/DRAM1

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
extern uint8_t dram0_pool[IDMA_BUFFER_SIZE_DRAM0];  // 40 KB pool
extern uint8_t dram1_pool[IDMA_BUFFER_SIZE_DRAM1];  // 40 KB pool

// Cache-mode padded input buffer (in system memory)
// Used by cache-mode kernels for edge padding
extern int8_t cache_padded_input[CACHE_PADDED_INPUT_SIZE];
//
//// Simple allocator state
//typedef struct {
//    size_t offset;
//    size_t size;
//} allocator_t;
//
//static allocator_t dram0_allocator = {0, sizeof(dram0_pool)};
//static allocator_t dram1_allocator = {0, sizeof(dram1_pool)};
//
//// Align size to 64-byte boundary
//static inline size_t align_size(size_t size) {
//    return (size + 63) & ~63;
//}
//
//// Allocate from DRAM pool
//static void* dram_alloc(allocator_t* alloc, uint8_t* pool, size_t size) {
//    size = align_size(size);
//
//    if (alloc->offset + size > alloc->size) {
//        printf("ERROR: DRAM allocation failed! Requested: %zu, Available: %zu\n",
//               size, alloc->size - alloc->offset);
//        return NULL;
//    }
//
//    void* ptr = pool + alloc->offset;
//    alloc->offset += size;
//    return ptr;
//}
//
//// Reset allocator (free all allocations)
//static void dram_reset(allocator_t* alloc) {
//    alloc->offset = 0;
//}
//
//// Allocate from DRAM0
//__attribute__((unused))
//static void* dram0_alloc(size_t size) {
//    return dram_alloc(&dram0_allocator, dram0_pool, size);
//}
//
//// Allocate from DRAM1
//__attribute__((unused))
//static void* dram1_alloc(size_t size) {
//    return dram_alloc(&dram1_allocator, dram1_pool, size);
//}
//
//// Reset both allocators
//__attribute__((unused))
//static void dram_reset_all(void) {
//    dram_reset(&dram0_allocator);
//    dram_reset(&dram1_allocator);
//}
//
//// Get current allocation status
//static inline size_t dram0_allocated(void) {
//    return dram0_allocator.offset;
//}
//
//static inline size_t dram1_allocated(void) {
//    return dram1_allocator.offset;
//}
//
//static inline size_t dram0_available(void) {
//    return dram0_allocator.size - dram0_allocator.offset;
//}
//
//static inline size_t dram1_available(void) {
//    return dram1_allocator.size - dram1_allocator.offset;
//}
//
//// Print allocation statistics
//static inline void dram_print_stats(void) {
//    printf("DRAM0: Used=%zu, Available=%zu, Total=%zu\n",
//           dram0_allocated(), dram0_available(), dram0_allocator.size);
//    printf("DRAM1: Used=%zu, Available=%zu, Total=%zu\n",
//           dram1_allocated(), dram1_available(), dram1_allocator.size);
//}

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
