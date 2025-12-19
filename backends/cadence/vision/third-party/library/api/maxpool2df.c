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

#include "api.h"
#include "common.h"

#if !HAVE_VFPU
DISCARD_FUN(void, maxpool2d_with_indices_j2x2_f32, (float32_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,int *restrict ptr_indices
  ,int inp_height ,int inp_width
  ,int out_height ,int out_width
  ,int32_t in_pitch_width, int32_t in_pitch_height
  ,int32_t out_pitch_width, int32_t out_pitch_height
  ,uint8_t kernel_height
  ,uint8_t kernel_width))

DISCARD_FUN(void, maxpool2d_j2x2_f32, (float32_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,int inp_height ,int inp_width
  ,int out_height ,int out_width
  ,int32_t in_pitch_width, int32_t in_pitch_height
  ,int32_t out_pitch_width, int32_t out_pitch_height
  ,uint8_t kernel_height
  ,uint8_t kernel_width))
#else
void maxpool2d_with_indices_j2x2_f32(float32_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,int *restrict ptr_indices
  ,int inp_height ,int inp_width
  ,int out_height ,int out_width
  ,int32_t in_pitch_width, int32_t in_pitch_height
  ,int32_t out_pitch_width, int32_t out_pitch_height
  ,uint8_t kernel_height
  ,uint8_t kernel_width)
{
  const int32_t out_increment = (((2 * IVP_SIMD_WIDTH) - kernel_width) / 2) + 1;

  int32_t x, y, kx, ky;
  int32_t remX, remXLoad;

  xb_vecN_2xf32* restrict pdvecOut;
  xb_vecN_2x32v* restrict pdvecIdx;
  xb_vecN_2xf32* restrict pdvecIn;

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2xf32 dvecMax1;
  xb_vecN_2xf32 dvecMax11, dvecMax12;
  xb_vecN_2xf32 dvecData11, dvecData12;
  xb_vecN_2x32v dvecKxIdx1, dvecKyIdx1;
  xb_vecN_2x32v dvecKyIdx11, dvecKyIdx12;
  xb_vecN_2x32v dvecIdx1;

  vboolN_2 dboolGT, dboolEq;
  vboolN_2 dboolkyIdxLT;
  xb_vecN_2x32v dvecGTKyIdx, dvecEQKyIdx;
  xb_vecN_2x32v dvecGTKxIdx, dvecEQKxIdx;

  vboolN_2 dvbKernelType = IVP_EQN_2X32((kernel_width % 2), 0);

  for (x = 0; x < out_width; x += out_increment) {
    remX = XT_MIN(out_width - x, out_increment);
    remXLoad = ((2 * (remX - 1) + kernel_width) > IVP_SIMD_WIDTH) ? 1 : 0;
    int32_t remXOffset = remXLoad * IVP_SIMD_WIDTH;

    for (y = 0; y < out_height; y++) {
      float* pOut = &ptr_out[y * out_pitch_width + x];
      int32_t* pIdx = &ptr_indices[y * out_pitch_width + x];
      const float* pSrc = ptr_inp + y * in_pitch_width * 2 + x * 2;
      pdvecIn = (xb_vecN_2xf32*) pSrc;

      // Initialize max values
      dvecMax1 = (MIN_FLT32);
      dvecMax11 = dvecMax12 = dvecMax1;

      // Initialize index tracking
      dvecKxIdx1 = 0;
      dvecKyIdx1 = 0;
      dvecKyIdx11 = dvecKyIdx12 = 0;

      // ========== KERNEL HEIGHT COMPARISONS ==========
      for (ky = 0; ky < kernel_height; ky++) {
        IVP_L2UN_2XF32_XP(dvecData11, pdvecIn, remXOffset * sizeof(float));
        IVP_L2UN_2XF32_XP(dvecData12, pdvecIn, (in_pitch_width - remXOffset) * sizeof(float));

        dboolGT = IVP_OGTN_2XF32(dvecData11, dvecMax11);
        dvecMax11 = IVP_MAXN_2XF32(dvecMax11, dvecData11);
        dvecKyIdx11 = IVP_MOVN_2X32T(ky, dvecKyIdx11, dboolGT);

        dboolGT = IVP_OGTN_2XF32(dvecData12, dvecMax12);
        dvecMax12 = IVP_MAXN_2XF32(dvecMax12, dvecData12);
        dvecKyIdx12 = IVP_MOVN_2X32T(ky, dvecKyIdx12, dboolGT);
      }

      IVP_DSELN_2XF32I(dvecMax12, dvecMax11, dvecMax12, dvecMax11, IVP_DSELI_32B_DEINTERLEAVE_1);
      IVP_DSELN_2X32I(dvecKyIdx12, dvecKyIdx11, dvecKyIdx12, dvecKyIdx11, IVP_DSELI_32B_DEINTERLEAVE_1);

      // ========== KERNEL WIDTH COMPARISONS ==========
      for (kx = 0; kx < kernel_width - 1; kx += 2) {
        // First comparison
        dboolEq = IVP_OEQN_2XF32(dvecMax11, dvecMax1);
        dboolGT = IVP_OGTN_2XF32(dvecMax11, dvecMax1);
        dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMax11);

        dvecGTKyIdx = IVP_MOVN_2X32T(dvecKyIdx11, dvecKyIdx1, dboolGT);
        dvecEQKyIdx = IVP_MOVN_2X32T(dvecKyIdx11, dvecKyIdx1, dboolEq);
        dvecKyIdx1 = IVP_MOVN_2X32T(IVP_MINN_2X32(dvecGTKyIdx, dvecEQKyIdx), dvecGTKyIdx, dboolEq);

        dvecGTKxIdx = IVP_MOVN_2X32T(kx, dvecKxIdx1, dboolGT);
        dvecEQKxIdx = IVP_MOVN_2X32T(kx, dvecKxIdx1, dboolEq);
        dboolkyIdxLT = IVP_LTN_2X32(dvecKyIdx1, dvecGTKyIdx);
        dvecKxIdx1 = IVP_MOVN_2X32T(IVP_MOVN_2X32T(dvecEQKxIdx, dvecGTKxIdx, dboolkyIdxLT), dvecGTKxIdx, dboolEq);

        dvecMax11 = IVP_SELN_2XF32I((MIN_FLT32), dvecMax11, IVP_SELI_32B_ROTATE_RIGHT_1);
        dvecKyIdx11 = IVP_SELN_2X32I(0, dvecKyIdx11, IVP_SELI_32B_ROTATE_RIGHT_1);

        // Second comparison
        dboolEq = IVP_OEQN_2XF32(dvecMax12, dvecMax1);
        dboolGT = IVP_OGTN_2XF32(dvecMax12, dvecMax1);
        dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMax12);

        dvecGTKyIdx = IVP_MOVN_2X32T(dvecKyIdx12, dvecKyIdx1, dboolGT);
        dvecEQKyIdx = IVP_MOVN_2X32T(dvecKyIdx12, dvecKyIdx1, dboolEq);
        dvecKyIdx1 = IVP_MOVN_2X32T(IVP_MINN_2X32(dvecGTKyIdx, dvecEQKyIdx), dvecGTKyIdx, dboolEq);

        dvecGTKxIdx = IVP_MOVN_2X32T((kx + 1), dvecKxIdx1, dboolGT);
        dvecEQKxIdx = IVP_MOVN_2X32T((kx + 1), dvecKxIdx1, dboolEq);
        dboolkyIdxLT = IVP_LTN_2X32(dvecKyIdx1, dvecGTKyIdx);
        dvecKxIdx1 = IVP_MOVN_2X32T(IVP_MOVN_2X32T(dvecEQKxIdx, dvecGTKxIdx, dboolkyIdxLT), dvecGTKxIdx, dboolEq);

        dvecMax12 = IVP_SELN_2XF32I((MIN_FLT32), dvecMax12, IVP_SELI_32B_ROTATE_RIGHT_1);
        dvecKyIdx12 = IVP_SELN_2X32I(0, dvecKyIdx12, IVP_SELI_32B_ROTATE_RIGHT_1);
      }

      // final comparison if kernel_width is odd
      xb_vecN_2xf32 dvecMaxTest = IVP_MOVN_2XF32T(dvecMax1, dvecMax11, dvbKernelType);

      dboolEq = IVP_OEQN_2XF32(dvecMaxTest, dvecMax1);
      dboolGT = IVP_OGTN_2XF32(dvecMaxTest, dvecMax1);
      dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMaxTest);

      dvecGTKyIdx = IVP_MOVN_2X32T(IVP_MOVN_2X32T(dvecKyIdx1, dvecKyIdx11, dvbKernelType), dvecKyIdx1, dboolGT);
      dvecEQKyIdx = IVP_MOVN_2X32T(IVP_MOVN_2X32T(dvecKyIdx1, dvecKyIdx11, dvbKernelType), dvecKyIdx1, dboolEq);
      dvecKyIdx1 = IVP_MOVN_2X32T(IVP_MINN_2X32(dvecGTKyIdx, dvecEQKyIdx), dvecGTKyIdx, dboolEq);

      dvecGTKxIdx = IVP_MOVN_2X32T(kx, dvecKxIdx1, dboolGT);
      dvecEQKxIdx = IVP_MOVN_2X32T(kx, dvecKxIdx1, dboolEq);
      dboolkyIdxLT = IVP_LTN_2X32(dvecKyIdx1, dvecGTKyIdx);
      dvecKxIdx1 = IVP_MOVN_2X32T(IVP_MOVN_2X32T(dvecEQKxIdx, dvecGTKxIdx, dboolkyIdxLT), dvecGTKxIdx, dboolEq);

      dvecIdx1 = IVP_ORN_2X32(IVP_SLLIN_2X32(dvecKyIdx1, 4), dvecKxIdx1);

      // ========== STORE OUTPUTS ==========
      // Store max values
      pdvecOut = (xb_vecN_2xf32*) pOut;
      IVP_SAVN_2XF32_XP(dvecMax1, vaOutData, pdvecOut, remX * sizeof(float));
      IVP_SAPOSN_2XF32_FP(vaOutData, pdvecOut);

      // Store indices
      pdvecIdx = (xb_vecN_2x32v*) pIdx;
      IVP_SAVN_2X32_XP(dvecIdx1, vaOutData, pdvecIdx, remX * sizeof(int32_t));
      IVP_SAPOSN_2X32_FP(vaOutData, pdvecIdx); 
    }
  }
}

void maxpool2d_j2x2_f32(float32_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,int inp_height ,int inp_width
  ,int out_height ,int out_width
  ,int32_t in_pitch_width, int32_t in_pitch_height
  ,int32_t out_pitch_width, int32_t out_pitch_height
  ,uint8_t kernel_height
  ,uint8_t kernel_width)
{
  const int32_t out_increment = (((2 * IVP_SIMD_WIDTH) - kernel_width) / 2) + 1;

  int32_t x, y, kx, ky;
  int32_t remX, remXLoad;

  xb_vecN_2xf32* restrict pdvecOut;
  xb_vecN_2xf32* restrict pdvecIn;

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2xf32 dvecMax1;
  xb_vecN_2xf32 dvecMax11, dvecMax12;
  xb_vecN_2xf32 dvecData11, dvecData12;

  vboolN_2 dvbKernelType = IVP_EQN_2X32((kernel_width % 2), 0);

  for (x = 0; x < out_width; x += out_increment) {
    remX = XT_MIN(out_width - x, out_increment);
    remXLoad = ((2 * (remX - 1) + kernel_width) > IVP_SIMD_WIDTH) ? 1 : 0;
    int32_t remXOffset = remXLoad * IVP_SIMD_WIDTH;

    for (y = 0; y < out_height; y++) {
      float* pOut = &ptr_out[y * out_pitch_width + x];
      const float* pSrc = ptr_inp + y * in_pitch_width * 2 + x * 2;
      pdvecIn = (xb_vecN_2xf32*) pSrc;

      // Initialize max values
      dvecMax1 = (MIN_FLT32);
      dvecMax11 = dvecMax12 = dvecMax1;

      // ========== KERNEL HEIGHT COMPARISONS ==========
      for (ky = 0; ky < kernel_height; ky++) {
        IVP_L2UN_2XF32_XP(dvecData11, pdvecIn, remXOffset * sizeof(float));
        IVP_L2UN_2XF32_XP(dvecData12, pdvecIn, (in_pitch_width - remXOffset) * sizeof(float));

        dvecMax11 = IVP_MAXN_2XF32(dvecMax11, dvecData11);
        dvecMax12 = IVP_MAXN_2XF32(dvecMax12, dvecData12);
      }

      IVP_DSELN_2XF32I(dvecMax12, dvecMax11, dvecMax12, dvecMax11, IVP_DSELI_32B_DEINTERLEAVE_1);

      // ========== KERNEL WIDTH COMPARISONS ==========
      for (kx = 0; kx < kernel_width - 1; kx += 2) {
        // First comparison
        dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMax11);
        dvecMax11 = IVP_SELN_2XF32I((MIN_FLT32), dvecMax11, IVP_SELI_32B_ROTATE_RIGHT_1);

        // Second comparison
        dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMax12);
        dvecMax12 = IVP_SELN_2XF32I((MIN_FLT32), dvecMax12, IVP_SELI_32B_ROTATE_RIGHT_1);
      }

      // final comparison if kernel_width is odd
      xb_vecN_2xf32 dvecMaxTest = IVP_MOVN_2XF32T(dvecMax1, dvecMax11, dvbKernelType);
      dvecMax1 = IVP_MAXN_2XF32(dvecMax1, dvecMaxTest);

      // ========== STORE OUTPUTS ==========
      // Store max values
      pdvecOut = (xb_vecN_2xf32*) pOut;
      IVP_SAVN_2XF32_XP(dvecMax1, vaOutData, pdvecOut, remX * sizeof(float));
      IVP_SAPOSN_2XF32_FP(vaOutData, pdvecOut);
    }
  }
}
#endif /* HAVE_VFPU */