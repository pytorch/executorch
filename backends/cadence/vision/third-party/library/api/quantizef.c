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
DISCARD_FUN(void, quantize_f32_asym8s, (int8_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,float32_t scale
  ,int zero_bias
  ,int N))
#else
void quantize_f32_asym8s(int8_t *restrict ptr_out
  ,const float32_t *restrict ptr_inp
  ,float32_t scale
  ,int zero_bias
  ,int N)
{
  // Inputs
  xb_vecN_2xf32 *p_i = (xb_vecN_2xf32 *)ptr_inp;
  xb_vecNx8 *p_o = (xb_vecNx8 *)ptr_out;
  float32_t one_by_scaleF = (float32_t) (1.0f / scale);
  float32_t one_by_scale = (one_by_scaleF > (float32_t) MAX_FLT32 ? (float32_t) MAX_FLT32 : (float32_t) (1.0f / scale));

  // Loop index
  int n;

  // Alignment variables
  valign al_i = IVP_LAN_2XF32_PP(p_i);
  valign al_o = IVP_ZALIGN();
  
  for (n = 0; n < (N >> LOG2_IVP_SIMD_WIDTH); n++)
  {
    xb_vecN_2xf32 inp1, inp2;
    xb_vecN_2xf32 inp1_scaled, inp2_scaled;
    xb_vecN_2xf32 out1, out2;
    xb_vecNx16 out;

    IVP_LAN_2XF32_IP(inp1, al_i, p_i);
    IVP_LAN_2XF32_IP(inp2, al_i, p_i);
    inp1_scaled = (float32_t) zero_bias;
    IVP_MULAN_2XF32(inp1_scaled, inp1, one_by_scale);
    inp2_scaled = (float32_t) zero_bias;
    IVP_MULAN_2XF32(inp2_scaled, inp2, one_by_scale);
    out1 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(inp1_scaled, (xb_vecN_2xf32) MAX_INT8), (xb_vecN_2xf32) MIN_INT8));
    out2 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(inp2_scaled, (xb_vecN_2xf32) MAX_INT8), (xb_vecN_2xf32) MIN_INT8));
    out = IVP_MOVNX16_FROMN_2X32(IVP_SELN_2X32I(out2, out1, IVP_SELI_EXTRACT_1_OF_2_OFF_0));
    IVP_SANX8S_IP(out, al_o, p_o);
  }
  if (N & (IVP_SIMD_WIDTH - 1))    // Check if there are remaining elements   
  {
    xb_vecN_2xf32 inp1, inp2;
    xb_vecN_2xf32 inp1_scaled, inp2_scaled;
    xb_vecN_2xf32 out1, out2;
    xb_vecNx16 out;

    IVP_LAVN_2XF32_XP(inp1, al_i, p_i, 4 * (N & (IVP_SIMD_WIDTH - 1)));
    IVP_LAVN_2XF32_XP(inp2, al_i, p_i, 4 * ((N & (IVP_SIMD_WIDTH - 1)) - (IVP_SIMD_WIDTH >> 1)));
    inp1_scaled = (float32_t) zero_bias;
    IVP_MULAN_2XF32(inp1_scaled, inp1, one_by_scale);
    inp2_scaled = (float32_t) zero_bias;
    IVP_MULAN_2XF32(inp2_scaled, inp2, one_by_scale);
    out1 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(inp1_scaled, (xb_vecN_2xf32) MAX_INT8), (xb_vecN_2xf32) MIN_INT8));
    out2 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(inp2_scaled, (xb_vecN_2xf32) MAX_INT8), (xb_vecN_2xf32) MIN_INT8));
    out = IVP_MOVNX16_FROMN_2X32(IVP_SELN_2X32I(out2, out1, IVP_SELI_EXTRACT_1_OF_2_OFF_0));
    IVP_SAVNX8S_XP(out, al_o, p_o, (N & (IVP_SIMD_WIDTH - 1)));
  }
  IVP_SAPOSNX8S_FP(al_o, p_o);
}
#endif