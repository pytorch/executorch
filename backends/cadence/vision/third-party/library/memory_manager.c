/*
 * memory_manager.c
 *
 *  Created on: Dec 8, 2025
 *      Author: Suraj Raut
 *
 *  Description: Definition of DRAM memory pools and local SRAM scratch buffer.
 *               These must be defined in exactly one compilation unit.
 */

#include "lib.h"
#include <xtensa/tie/xt_ivpn.h>  // For XCHAL_IVPN_SIMD_WIDTH

// Memory pools placed in specific DRAM sections
// These are the actual storage for the DRAM pools
__attribute__((section(".dram0.data"))) __attribute__((aligned(64*2))) 
uint8_t dram0_pool[IDMA_BUFFER_SIZE_DRAM0];  // 40 KB pool in DRAM0

__attribute__((section(".dram1.data"))) __attribute__((aligned(64*2))) 
uint8_t dram1_pool[IDMA_BUFFER_SIZE_DRAM1];  // 40 KB pool in DRAM1

// Cache-mode padded input buffer (in system memory)
// Used by cache-mode kernels for edge padding before convolution
// This buffer is accessed through the processor's data cache
__attribute__((aligned(64*2)))
int8_t cache_padded_input[CACHE_PADDED_INPUT_SIZE];  // 1 MB max

/**
 * Allocate DRAM buffer with SIMD alignment
 */
int8_t* allocate_dram_buffer(int size, int dram_bank, int* dram0_used, int* dram1_used) {
    int8_t* ptr;
    int aligned_size = (size + (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) & ~(2 * XCHAL_IVPN_SIMD_WIDTH - 1);
    
    if (dram_bank == 0) {
        ptr = (int8_t*)(dram0_pool + *dram0_used);
        *dram0_used += aligned_size;
    } else {
        ptr = (int8_t*)(dram1_pool + *dram1_used);
        *dram1_used += aligned_size;
    }
    
    return ptr;
}
