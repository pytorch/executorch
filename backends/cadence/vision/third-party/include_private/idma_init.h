#ifndef __IDMA__INIT_H__
#define __IDMA__INIT_H__

#include "dtypes.h"
#include "common.h"

#define IDMA_BUFF_SIZE 16384 // 16 kb DRAM storage. Assume 4 buffers (2 input and 2 output)

#ifndef PLACE_IN_DRAM0
	#define PLACE_IN_DRAM0 __attribute__ ((aligned(2*IVP_SIMD_WIDTH), section(".dram0.data")))
#endif

#ifndef PLACE_IN_DRAM1
	#define PLACE_IN_DRAM1 __attribute__ ((aligned(2*IVP_SIMD_WIDTH), section(".dram1.data")))
#endif

float32_t data_dram0[IDMA_BUFF_SIZE / 2] PLACE_IN_DRAM0;
float32_t data_dram1[IDMA_BUFF_SIZE / 2] PLACE_IN_DRAM1;

float32_t *inpData[2] = {&data_dram0[0], &data_dram1[0]};
float32_t *outData[2] = {&data_dram0[IDMA_BUFF_SIZE / 4], &data_dram1[IDMA_BUFF_SIZE / 4]};

IDMA_BUFFER_DEFINE(buffer_idma_ch0, 1, IDMA_2D_DESC);
IDMA_BUFFER_DEFINE(buffer_idma_ch1, 1, IDMA_2D_DESC);

idma_buffer_t * descbuf[] = {
  buffer_idma_ch0,
  buffer_idma_ch1,
};

#endif // __IDMA__INIT_H__