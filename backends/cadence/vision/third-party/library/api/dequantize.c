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
DISCARD_FUN(void, dequantize_asym8s_f32, (float32_t *restrict ptr_out
  ,const int8_t *restrict ptr_inp
  ,float32_t scale
  ,int zero_bias
  ,int N))
#else
void dequantize_asym8s_f32(float32_t *restrict ptr_out
  ,const int8_t *restrict ptr_inp
  ,float32_t scale
  ,int zero_bias
  ,int N)
{
  // Inputs
  xb_vecNx8 *p_i = (xb_vecNx8 *)ptr_inp;
  xb_vecN_2xf32 *p_o = (xb_vecN_2xf32 *)ptr_out;

  // Loop index
  int n;

  // Alignment variables
  valign al_i = IVP_LANX8S_PP(p_i);
  valign al_o = IVP_ZALIGN();
  
  for (n = 0; n < (N >> LOG2_IVP_SIMD_WIDTH); n++)
  {
    xb_vecNx16 inp;
    xb_vecN_2x32v inp1_bias, inp2_bias;
    xb_vecN_2xf32 out1, out2;

    IVP_LANX8S_XP(inp, al_i, p_i, IVP_SIMD_WIDTH);

    inp1_bias = IVP_UNPKSNX16_L(inp);
    inp2_bias = IVP_UNPKSNX16_H(inp);

    inp1_bias = IVP_SUBN_2X32(inp1_bias, (xb_vecN_2x32v) zero_bias);
    out1 = IVP_MULN_2XF32(scale, (xb_vecN_2xf32) inp1_bias);

    inp2_bias = IVP_SUBN_2X32(inp2_bias, (xb_vecN_2x32v) zero_bias);
    out2 = IVP_MULN_2XF32(scale, (xb_vecN_2xf32) inp2_bias);

    IVP_SAN_2XF32_IP(out1, al_o, p_o);
    IVP_SAN_2XF32_IP(out2, al_o, p_o);
  }
  if (N & (IVP_SIMD_WIDTH - 1)) // Check if there are remaining elements
  {
    xb_vecNx16 inp;
    xb_vecN_2x32v inp1_bias, inp2_bias;
    xb_vecN_2xf32 out1, out2;

    IVP_LANX8S_XP(inp, al_i, p_i, N & (IVP_SIMD_WIDTH - 1));

    inp1_bias = IVP_UNPKSNX16_L(inp);
    inp2_bias = IVP_UNPKSNX16_H(inp);

    inp1_bias = IVP_SUBN_2X32(inp1_bias, (xb_vecN_2x32v) zero_bias);
    out1 = IVP_MULN_2XF32(scale, (xb_vecN_2xf32) inp1_bias);

    inp2_bias = IVP_SUBN_2X32(inp2_bias, (xb_vecN_2x32v) zero_bias);
    out2 = IVP_MULN_2XF32(scale, (xb_vecN_2xf32) inp2_bias);

    IVP_SAVN_2XF32_XP(out1, al_o, p_o, 4 * (N & (IVP_SIMD_WIDTH - 1)));
    IVP_SAVN_2XF32_XP(out2, al_o, p_o, 4 * ((N & (IVP_SIMD_WIDTH - 1)) - (IVP_SIMD_WIDTH >> 1)));
  }
  IVP_SAPOSN_2XF32_FP(al_o, p_o);
}
#endif