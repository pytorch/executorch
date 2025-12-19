/* ------------------------------------------------------------------------ */
/* Copyright (c) 2024 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ('Cadence    */
/* Libraries') are the copyrighted works of Cadence Design Systems Inc.     */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */

#ifndef __LIB_H__
#define __LIB_H__

#include "dtypes.h"
#include "api.h"
#include <stdio.h>

#if defined COMPILER_XTENSA

#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_ivpn.h>
#define IVP_SIMD_WIDTH XCHAL_IVPN_SIMD_WIDTH

// Performance measurement macros
#define XTPERF_PRINTF(...) printf(__VA_ARGS__)
#define TIME_DECL(test) long start_time_##test, end_time_##test;
#define TIME_START(test) { start_time_##test = 0;   XT_WSR_CCOUNT(0); }
#define TIME_END(test) { end_time_##test = XT_RSR_CCOUNT(); }
#define TIME_DISPLAY(test, opcnt, opname) { long long cycles_##test = end_time_##test - start_time_##test; \
		XTPERF_PRINTF("PERF_LOG : %s : %d : %s : %lld : cycles : %.2f : %s/cycle : %.2f : cycles/%s\n", \
		       #test, opcnt, opname, cycles_##test, cycles_##test == 0 ? 0 : (double)(opcnt)/cycles_##test, \
           opname, cycles_##test == 0 ? 0 : 1/((double)(opcnt)/cycles_##test), opname); }


// IDMA Initializations and declarations
#if XCHAL_HAVE_IDMA
#ifndef IDMA_USE_MULTICHANNEL
  #define IDMA_USE_MULTICHANNEL 1
#endif
#ifndef CHL_MAX
  #define CHL_MAX 2
#endif
#include <xtensa/idma.h>
#endif

#ifndef DRAM0_BUFF_SIZE // To be defined at compile time
  #error "DRAM0_BUFF_SIZE not defined"
#endif

#ifndef DRAM1_BUFF_SIZE // To be defined at compile time
  #error "DRAM1_BUFF_SIZE not defined"
#endif

#ifndef PLACE_IN_DRAM0
	#define PLACE_IN_DRAM0 __attribute__ ((aligned(2*IVP_SIMD_WIDTH), section(".dram0.data")))
#endif

#ifndef PLACE_IN_DRAM1
	#define PLACE_IN_DRAM1 __attribute__ ((aligned(2*IVP_SIMD_WIDTH), section(".dram1.data")))
#endif

extern void *ptr_dram0;
extern void *ptr_dram1;

extern idma_buffer_t buffer_idma_ch_2d[];
extern idma_buffer_t buffer_idma_ch_3d[];

#endif // COMPILER_XTENSA

#endif // __LIB_H__