
#include "api.h"
#include "common.h"
#include <math.h>

void vrelU_quantized(
    int8_t* restrict ptr_out,
    const int8_t* restrict ptr_inp,
    int32_t in_zero_point,
    int32_t out_zero_point,
    float32_t out_scale,
    int N)
{
  // Pointers
  xb_vecNx8 *p_i = (xb_vecNx8 *)ptr_inp;
  xb_vecNx8 *p_o = (xb_vecNx8 *)ptr_out;

  // Loop index
  int n;

  // Alignment variables
  valign al_i = IVP_LANX8S_PP(p_i);
  valign al_o = IVP_ZALIGN();

  // Constants
  xb_vecN_2x32v zero_vec = 0;
  xb_vecN_2x32v in_zp_vec = (xb_vecN_2x32v)in_zero_point;
  xb_vecN_2xf32 out_zp_f32 = (xb_vecN_2xf32)(float32_t)out_zero_point;
  xb_vecN_2xf32 min_val = (xb_vecN_2xf32)(-128.0f);
  xb_vecN_2xf32 max_val = (xb_vecN_2xf32)(127.0f);

  for (n = 0; n < (N >> LOG2_IVP_SIMD_WIDTH); n++)
  {
    xb_vecNx16 inp;
    xb_vecN_2x32v temp1, temp2;
    xb_vecN_2xf32 float1, float2;
    xb_vecN_2xf32 result1, result2;
    xb_vecNx16 out;

    // Load int8 → sign-extend to 16-bit
    IVP_LANX8S_XP(inp, al_i, p_i, IVP_SIMD_WIDTH);

    // Unpack 16-bit → two 32-bit vectors (16 elements each)
    temp1 = IVP_UNPKSNX16_L(inp);
    temp2 = IVP_UNPKSNX16_H(inp);

    // Integer operations: SUB in_zero_point
    temp1 = IVP_SUBN_2X32(temp1, in_zp_vec);
    temp2 = IVP_SUBN_2X32(temp2, in_zp_vec);

    // ReLU: MAX(temp, 0)
    temp1 = IVP_MAXN_2X32(temp1, zero_vec);
    temp2 = IVP_MAXN_2X32(temp2, zero_vec);

    // Convert int32 → float32 (implicit cast)
    float1 = (xb_vecN_2xf32)temp1;
    float2 = (xb_vecN_2xf32)temp2;

    // FMA: out_zero_point + temp * out_scale
    result1 = out_zp_f32;
    IVP_MULAN_2XF32(result1, float1, out_scale);
    result2 = out_zp_f32;
    IVP_MULAN_2XF32(result2, float2, out_scale);

    // Clamp to [-128, 127] and round to nearest integer
    result1 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(result1, max_val), min_val));
    result2 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(result2, max_val), min_val));

    // Pack float → int16 → int8 (no explicit conversion needed)
    out = IVP_MOVNX16_FROMN_2X32(IVP_SELN_2X32I(result2, result1, IVP_SELI_EXTRACT_1_OF_2_OFF_0));
    IVP_SANX8S_IP(out, al_o, p_o);
  }

  // Handle remaining elements (tail)
  if (N & (IVP_SIMD_WIDTH - 1))
  {
    xb_vecNx16 inp;
    xb_vecN_2x32v temp1, temp2;
    xb_vecN_2xf32 float1, float2;
    xb_vecN_2xf32 result1, result2;
    xb_vecNx16 out;

    IVP_LANX8S_XP(inp, al_i, p_i, N & (IVP_SIMD_WIDTH - 1));

    temp1 = IVP_UNPKSNX16_L(inp);
    temp2 = IVP_UNPKSNX16_H(inp);

    temp1 = IVP_SUBN_2X32(temp1, in_zp_vec);
    temp2 = IVP_SUBN_2X32(temp2, in_zp_vec);

    temp1 = IVP_MAXN_2X32(temp1, zero_vec);
    temp2 = IVP_MAXN_2X32(temp2, zero_vec);

    float1 = (xb_vecN_2xf32)temp1;
    float2 = (xb_vecN_2xf32)temp2;

    result1 = out_zp_f32;
    IVP_MULAN_2XF32(result1, float1, out_scale);
    result2 = out_zp_f32;
    IVP_MULAN_2XF32(result2, float2, out_scale);

    // Clamp to [-128, 127] and round to nearest integer
    result1 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(result1, max_val), min_val));
    result2 = IVP_FIRINTN_2XF32(IVP_MAXN_2XF32(IVP_MINN_2XF32(result2, max_val), min_val));

    // Pack float → int16 → int8 (no explicit conversion needed)
    out = IVP_MOVNX16_FROMN_2X32(IVP_SELN_2X32I(result2, result1, IVP_SELI_EXTRACT_1_OF_2_OFF_0));
    IVP_SAVNX8S_XP(out, al_o, p_o, (N & (IVP_SIMD_WIDTH - 1)));
  }

  IVP_SAPOSNX8S_FP(al_o, p_o);
}
