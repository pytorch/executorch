/*
 * Global variable definitions for lib.h
 * This file contains the actual definitions to avoid multiple definition errors
 */

#include "../include/lib.h"

#if defined COMPILER_XTENSA

uint8_t dram0_buffer[DRAM0_BUFF_SIZE] PLACE_IN_DRAM0;
uint8_t dram1_buffer[DRAM1_BUFF_SIZE] PLACE_IN_DRAM1;

void *ptr_dram0 = (void *)dram0_buffer;
void *ptr_dram1 = (void *)dram1_buffer;

IDMA_BUFFER_DEFINE(buffer_idma_ch_2d, 2 * CHL_MAX, IDMA_2D_DESC);
IDMA_BUFFER_DEFINE(buffer_idma_ch_3d, 2 * CHL_MAX, IDMA_64B_DESC);

#endif // COMPILER_XTENSA
