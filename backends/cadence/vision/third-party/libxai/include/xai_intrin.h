/*
 * Copyright (c) 2013-2018 Tensilica Inc. ALL RIGHTS RESERVED.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __XAI_INTRIN_H__
#define __XAI_INTRIN_H__

#include <xtensa/tie/xt_ivpn.h>
#include <xtensa/tie/xt_misc.h>

#ifndef XCHAL_HAVE_VISION
#  define XCHAL_HAVE_VISION  0
#endif

#if ((XCHAL_HW_REL_LX8 == 1) && (XCHAL_VISION_SIMD16 == 32))
#define IS_VISION_130
#endif

////////// CSTUBS workarounds

#if defined(_MSC_VER) && !XCHAL_HAVE_VISION
#   undef IVP_ABSSUBNX16
#   define IVP_ABSSUBNX16(a, b)  IVP_MAXNX16(IVP_SUBNX16(b, a), IVP_SUBNX16(a, b))
#endif

#if !defined(__XCC__) && !XCHAL_HAVE_VISION
typedef vselN  _xai_intrin_private_xb_vselN;
#   undef IVP_SQZN
#   define IVP_SQZN(a, b, c)    do { _xai_intrin_private_xb_vselN _sqzntmp; CSTUB_(_TIE_xt_ivp32_IVP_SQZN) (_sqzntmp, b, c); a = _sqzntmp; } while (0)
#   undef IVP_UNSQZN
#   define IVP_UNSQZN(a, b, c)  do { _xai_intrin_private_xb_vselN _sqzntmp; CSTUB_(_TIE_xt_ivp32_IVP_UNSQZN) (_sqzntmp, b, c); a = _sqzntmp; } while (0)
#endif

#if !defined(__XCC__) && XCHAL_HAVE_VISION
#if 0
#undef IVP_SCATTERNX8U
#undef IVP_SCATTERNX8UT
#define IVP_SCATTERNX8U(val__, ptr__, offs__)                        \
  {                                                                  \
    vboolN mask       = IVP_LTNX16(0, 1);                            \
    xb_vecNx16 mask16 = IVP_MOVNX16T(1, 0, mask);                    \
    xb_vecNx16U offs1 = (offs__);                                    \
    xb_vecNx16 val1   = val__;                                       \
    for (int i = 0; i < 32; i++)                                     \
    {                                                                \
      int v = IVP_MOVAVU16(val1);                                    \
      int o = IVP_MOVAVU16(offs1);                                   \
      int m = IVP_MOVAVU16(mask16);                                  \
      if (m) { *((uint8_t *) (ptr__) + o) = v; }                     \
      val1   = IVP_SELNX16I(0, val1, IVP_SELI_16B_ROTATE_RIGHT_1);   \
      mask16 = IVP_SELNX16I(0, mask16, IVP_SELI_16B_ROTATE_RIGHT_1); \
      offs1  = IVP_SELNX16I(0, offs1, IVP_SELI_16B_ROTATE_RIGHT_1);  \
    }                                                                \
  }

#define IVP_SCATTERNX8UT(val__, ptr__, offs__, mask__)               \
  {                                                                  \
    xb_vecNx16 mask16 = IVP_MOVNX16T(1, 0, (mask__));                \
    xb_vecNx16 val    = (val__);                                     \
    xb_vecNx16 off    = (offs__);                                    \
    for (int i = 0; i < 32; i++)                                     \
    {                                                                \
      int v = IVP_MOVAVU16(val);                                     \
      int o = IVP_MOVAVU16(off);                                     \
      int m = IVP_MOVAVU16(mask16);                                  \
      if (m) { *(((uint8_t *) ptr__) + o) = v; }                     \
      val    = IVP_SELNX16I(0, val, IVP_SELI_16B_ROTATE_RIGHT_1);    \
      mask16 = IVP_SELNX16I(0, mask16, IVP_SELI_16B_ROTATE_RIGHT_1); \
      off    = IVP_SELNX16I(0, off, IVP_SELI_16B_ROTATE_RIGHT_1);    \
    }                                                                \
  }

#undef IVP_SCATTERN_2X32
#undef IVP_SCATTERN_2X32T
#define IVP_SCATTERN_2X32(val__, ptr__, offs__)                                                                      \
  {                                                                                                                  \
    vboolN_2 mask        = IVP_LTN_2X32(0, 1);                                                                       \
    xb_vecN_2x32v mask32 = IVP_MOVN_2X32T(1, 0, mask);                                                               \
    xb_vecN_2x32v offs1  = IVP_SRLIN_2X32(offs__, 2);                                                                \
    xb_vecN_2x32v val1   = val__;                                                                                    \
    for (int i = 0; i < 16; i++)                                                                                     \
    {                                                                                                                \
      int v = IVP_MOVAV32(val1);                                                                                     \
      int o = IVP_MOVAV32(offs1);                                                                                    \
      int m = IVP_MOVAV32(mask32);                                                                                   \
      if (m) { *((ptr__) + o) = v; }                                                                                 \
      val1   = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(val1), IVP_SELI_32B_ROTATE_RIGHT_1));   \
      mask32 = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(mask32), IVP_SELI_32B_ROTATE_RIGHT_1)); \
      offs1  = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(offs1), IVP_SELI_32B_ROTATE_RIGHT_1));  \
    }                                                                                                                \
  }

#define IVP_SCATTERN_2X32T(val__, ptr__, offs__, mask__)                                                             \
  {                                                                                                                  \
    xb_vecN_2x32v mask32 = IVP_MOVN_2X32T(1, 0, mask__);                                                             \
    xb_vecN_2x32v offs1  = IVP_SRLIN_2X32(offs__, 2);                                                                \
    xb_vecN_2x32v val1   = val__;                                                                                    \
    for (int i = 0; i < 16; i++)                                                                                     \
    {                                                                                                                \
      int v = IVP_MOVAV32(val1);                                                                                     \
      int o = IVP_MOVAV32(offs1);                                                                                    \
      int m = IVP_MOVAV32(mask32);                                                                                   \
      if (m) { *((ptr__) + o) = v; }                                                                                 \
      val1   = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(val1), IVP_SELI_32B_ROTATE_RIGHT_1));   \
      mask32 = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(mask32), IVP_SELI_32B_ROTATE_RIGHT_1)); \
      offs1  = IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(0, IVP_MOVNX16_FROMN_2X32(offs1), IVP_SELI_32B_ROTATE_RIGHT_1));  \
    }                                                                                                                \
  }

#undef IVP_SCATTER2NX8U_L
#undef IVP_SCATTER2NX8UT_L
#define IVP_SCATTER2NX8U_L(val__, ptr__, offs__)                   \
  {                                                                \
    vbool2N mask      = IVP_LT2NX8(0, 1);                          \
    xb_vec2Nx8 mask8  = IVP_MOV2NX8T(1, 0, mask);                  \
    xb_vecNx16U offs1 = (offs__);                                  \
    xb_vec2Nx8 val1   = val__;                                     \
    for (int i = 0; i < 32; i++)                                   \
    {                                                              \
      int v = IVP_MOVAVU8(val1);                                   \
      int o = IVP_MOVAVU16(offs1);                                 \
      int m = IVP_MOVAVU8(mask8);                                  \
      if (m) { *((uint8_t *) (ptr__) + o) = v; }                   \
      val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_1);   \
      mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_1);  \
      offs1 = IVP_SELNX16I(0, offs1, IVP_SELI_16B_ROTATE_RIGHT_1); \
    }                                                              \
  }

#define IVP_SCATTER2NX8UT_L(val__, ptr__, offs__, mask__)          \
  {                                                                \
    vbool2N mask      = mask__;                                    \
    xb_vec2Nx8 mask8  = IVP_MOV2NX8T(1, 0, mask);                  \
    xb_vecNx16U offs1 = (offs__);                                  \
    xb_vec2Nx8 val1   = val__;                                     \
    for (int i = 0; i < 32; i++)                                   \
    {                                                              \
      int v = IVP_MOVAVU8(val1);                                   \
      int o = IVP_MOVAVU16(offs1);                                 \
      int m = IVP_MOVAVU8(mask8);                                  \
      if (m) { *((uint8_t *) (ptr__) + o) = v; }                   \
      val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_1);   \
      mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_1);  \
      offs1 = IVP_SELNX16I(0, offs1, IVP_SELI_16B_ROTATE_RIGHT_1); \
    }                                                              \
  }

#undef IVP_SCATTER2NX8U_H
#undef IVP_SCATTER2NX8UT_H
#define IVP_SCATTER2NX8U_H(val__, ptr__, offs__)                   \
  {                                                                \
    vbool2N mask      = IVP_LT2NX8(0, 1);                          \
    xb_vec2Nx8 mask8  = IVP_MOV2NX8T(1, 0, mask);                  \
    xb_vecNx16U offs1 = (offs__);                                  \
    xb_vec2Nx8 val1   = val__;                                     \
                                                                   \
    val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_32);    \
    mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_32);   \
    for (int i = 0; i < 32; i++)                                   \
    {                                                              \
      int v = IVP_MOVAVU8(val1);                                   \
      int o = IVP_MOVAVU16(offs1);                                 \
      int m = IVP_MOVAVU8(mask8);                                  \
      if (m) { *((uint8_t *) (ptr__) + o) = v; }                   \
      val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_1);   \
      mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_1);  \
      offs1 = IVP_SELNX16I(0, offs1, IVP_SELI_16B_ROTATE_RIGHT_1); \
    }                                                              \
  }

#define IVP_SCATTER2NX8UT_H(val__, ptr__, offs__, mask__)          \
  {                                                                \
    vbool2N mask      = mask__;                                    \
    xb_vec2Nx8 mask8  = IVP_MOV2NX8T(1, 0, mask);                  \
    xb_vecNx16U offs1 = (offs__);                                  \
    xb_vec2Nx8 val1   = val__;                                     \
                                                                   \
    val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_32);    \
    mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_32);   \
    for (int i = 0; i < 32; i++)                                   \
    {                                                              \
      int v = IVP_MOVAVU8(val1);                                   \
      int o = IVP_MOVAVU16(offs1);                                 \
      int m = IVP_MOVAVU8(mask8);                                  \
      if (m) { *((uint8_t *) (ptr__) + o) = v; }                   \
      val1  = IVP_SEL2NX8I(0, val1, IVP_SELI_8B_ROTATE_RIGHT_1);   \
      mask8 = IVP_SEL2NX8I(0, mask8, IVP_SELI_8B_ROTATE_RIGHT_1);  \
      offs1 = IVP_SELNX16I(0, offs1, IVP_SELI_16B_ROTATE_RIGHT_1); \
    }                                                              \
  }

#undef IVP_GATHERNX8UT_V
#define IVP_GATHERNX8UT_V(pdst, offs, mask, dly)  IVP_MOVNX16T(IVP_GATHERNX8U_V((pdst), (offs), (dly)), 0, mask)

#undef IVP_GATHERNX16T_V
#define IVP_GATHERNX16T_V(pdst, offs, mask, dly)  IVP_MOVNX16T(IVP_GATHERNX16_V((pdst), (offs), (dly)), 0, mask)

#undef IVP_GATHERN_2X32T_V
#define IVP_GATHERN_2X32T_V(pdst, offs, mask, dly)  IVP_MOVN_2X32T(IVP_GATHERN_2X32_V((pdst), (offs), (dly)), 0, mask)
#endif // #if 0
#endif //!defined(__XCC__) && XCHAL_HAVE_VISION

#if XCHAL_VISION_QUAD_MAC_TYPE == 0
#ifndef IVP_MULQA2N8XR8
#define IVP_MULQA2N8XR8(_dacc_, _dvec3_, _dvec2_, _dvec1_, _dvec0_, _scalar32_)  {            \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_))); \
    IVP_MULA2NX8(_dacc_, _dvec0_, IVP_REP2NX8(dvecS, 0));                                     \
    IVP_MULA2NX8(_dacc_, _dvec1_, IVP_REP2NX8(dvecS, 1));                                     \
    IVP_MULA2NX8(_dacc_, _dvec2_, IVP_REP2NX8(dvecS, 2));                                     \
    IVP_MULA2NX8(_dacc_, _dvec3_, IVP_REP2NX8(dvecS, 3));                                     \
}
#endif

#ifndef IVP_MULUSQA2N8XR8
#define IVP_MULUSQA2N8XR8(_dacc_, _dvec3_, _dvec2_, _dvec1_, _dvec0_, _scalar32_)  {          \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_))); \
    IVP_MULUSA2NX8(_dacc_, _dvec0_, IVP_REP2NX8(dvecS, 0));                                   \
    IVP_MULUSA2NX8(_dacc_, _dvec1_, IVP_REP2NX8(dvecS, 1));                                   \
    IVP_MULUSA2NX8(_dacc_, _dvec2_, IVP_REP2NX8(dvecS, 2));                                   \
    IVP_MULUSA2NX8(_dacc_, _dvec3_, IVP_REP2NX8(dvecS, 3));                                   \
}
#endif

#if 0 // Currently disabled as there is no usecase. Kept it so that it can be used in future if required.
#ifndef IVP_MULSUQ2N8XR8
static inline xb_vec2Nx24 IVP_MULSUQ2N8XR8(xb_vec2Nx8 _dvec3_, xb_vec2Nx8 _dvec2_, xb_vec2Nx8 _dvec1_, xb_vec2Nx8 _dvec0_, int32_t _scalar32_)
{
  xb_vec2Nx24 _dacc_;
  xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));
  _dacc_ = IVP_MULUS2NX8(IVP_REP2NX8(dvecS, 0), _dvec0_);
  IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 1), _dvec1_);
  IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 2), _dvec2_);
  IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 3), _dvec3_);
  return(_dacc_);
}
#endif

#ifndef IVP_MULSUQA2N8XR8
#define IVP_MULSUQA2N8XR8(_dacc_, _dvec3_, _dvec2_, _dvec1_, _dvec0_, _scalar32_)  {          \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_))); \
    IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 0), _dvec0_);                                   \
    IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 1), _dvec1_);                                   \
    IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 2), _dvec2_);                                   \
    IVP_MULUSA2NX8(_dacc_, IVP_REP2NX8(dvecS, 3), _dvec3_);                                   \
}
#endif

#ifndef IVP_MULUUQA2N8XR8
#define IVP_MULUUQA2N8XR8(_dacc_, _dvec3_, _dvec2_, _dvec1_, _dvec0_, _scalar32_)  {          \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_))); \
    IVP_MULUUA2NX8(_dacc_, _dvec0_, IVP_REP2NX8(dvecS, 0));                                   \
    IVP_MULUUA2NX8(_dacc_, _dvec1_, IVP_REP2NX8(dvecS, 1));                                   \
    IVP_MULUUA2NX8(_dacc_, _dvec2_, IVP_REP2NX8(dvecS, 2));                                   \
    IVP_MULUUA2NX8(_dacc_, _dvec3_, IVP_REP2NX8(dvecS, 3));                                   \
}
#endif
#endif

#ifndef IVP_MUL4TA2N8XR8
#define IVP_MUL4TA2N8XR8(_dacc_, _dvec1_, _dvec0_, _scalar32_)  {                                            \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));                \
    IVP_MULA2NX8(_dacc_, _dvec0_, IVP_REP2NX8(dvecS, 0));                                                    \
    IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_1), IVP_REP2NX8(dvecS, 1)); \
    IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_2), IVP_REP2NX8(dvecS, 2)); \
    IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_3), IVP_REP2NX8(dvecS, 3)); \
}
#endif

#ifndef IVP_MULUS4TA2N8XR8
#define IVP_MULUS4TA2N8XR8(_dacc_, _dvec1_, _dvec0_, _scalar32_)  {                                             \
    xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));                   \
    IVP_MULUSA2NX8(_dacc_, _dvec0_, IVP_REP2NX8(dvecS, 0));                                                     \
    IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_1), IVP_REP2NX8(dvecS, 1)); \
    IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_2), IVP_REP2NX8(dvecS, 2)); \
    IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_3), IVP_REP2NX8(dvecS, 3)); \
}
#endif

#ifndef IVP_MUL4T2N8XR8
static inline xb_vec2Nx24 IVP_MUL4T2N8XR8(xb_vec2Nx8U _dvec1_, xb_vec2Nx8U _dvec0_, int _scalar32_)
{
  xb_vec2Nx24 _dacc_;
  xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));
  _dacc_ = IVP_MUL2NX8(_dvec0_, IVP_REP2NX8(dvecS, 0));
  IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_1), IVP_REP2NX8(dvecS, 1));
  IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_2), IVP_REP2NX8(dvecS, 2));
  IVP_MULA2NX8(_dacc_, IVP_SEL2NX8I(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_3), IVP_REP2NX8(dvecS, 3));
  return(_dacc_);
}
#endif

#ifndef IVP_MULUS4T2N8XR8
static inline xb_vec2Nx24 IVP_MULUS4T2N8XR8(xb_vec2Nx8U _dvec1_, xb_vec2Nx8U _dvec0_, int _scalar32_)
{
  xb_vec2Nx24 _dacc_;
  xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));
  _dacc_ = IVP_MULUS2NX8(_dvec0_, IVP_REP2NX8(dvecS, 0));
  IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_1), IVP_REP2NX8(dvecS, 1));
  IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_2), IVP_REP2NX8(dvecS, 2));
  IVP_MULUSA2NX8(_dacc_, IVP_SEL2NX8UI(_dvec1_, _dvec0_, IVP_SELI_8B_ROTATE_RIGHT_3), IVP_REP2NX8(dvecS, 3));
  return(_dacc_);
}
#endif

#ifndef IVP_MULQ2N8XR8
static inline xb_vec2Nx24 IVP_MULQ2N8XR8(xb_vec2Nx8 _dvec3_, xb_vec2Nx8 _dvec2_, xb_vec2Nx8 _dvec1_, xb_vec2Nx8 _dvec0_, int32_t _scalar32_)
{
  xb_vec2Nx24 _dacc_;
  xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));
  _dacc_ = IVP_MUL2NX8(_dvec0_, IVP_REP2NX8(dvecS, 0));
  IVP_MULA2NX8(_dacc_, _dvec1_, IVP_REP2NX8(dvecS, 1));
  IVP_MULA2NX8(_dacc_, _dvec2_, IVP_REP2NX8(dvecS, 2));
  IVP_MULA2NX8(_dacc_, _dvec3_, IVP_REP2NX8(dvecS, 3));
  return(_dacc_);
}
#endif

#ifndef IVP_MULUSQ2N8XR8
static inline xb_vec2Nx24 IVP_MULUSQ2N8XR8(xb_vec2Nx8U _dvec3_, xb_vec2Nx8U _dvec2_, xb_vec2Nx8U _dvec1_, xb_vec2Nx8U _dvec0_, int32_t _scalar32_)
{
  xb_vec2Nx24 _dacc_;
  xb_vec2Nx8 dvecS = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(_scalar32_)));
  _dacc_ = IVP_MULUS2NX8(_dvec0_, IVP_REP2NX8(dvecS, 0));
  IVP_MULUSA2NX8(_dacc_, _dvec1_, IVP_REP2NX8(dvecS, 1));
  IVP_MULUSA2NX8(_dacc_, _dvec2_, IVP_REP2NX8(dvecS, 2));
  IVP_MULUSA2NX8(_dacc_, _dvec3_, IVP_REP2NX8(dvecS, 3));
  return(_dacc_);
}
#endif
#endif //#if XCHAL_VISION_QUAD_MAC_TYPE == 0

#if XCHAL_HAVE_SUPERGATHER == 0

#ifdef IVP_GATHERANX8S
#undef IVP_GATHERANX8S
static inline xb_vecNx16 IVP_GATHERANX8S(const signed char * _base, xb_vecNx16U _offsets)
{
  const signed char *_basePtr = _base;          \
  xb_vecNx16U _offsetsVec     = _offsets;       \
  xb_vecNx16 _dataVec         = (xb_vecNx16) 0; \
  int _i;                                       \
  for (_i = 0; _i < 32; _i++)
  {
                                                                                                 \
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                           \
    xb_int8 gdata         = IVP_LS2NX8_X(_basePtr, offset);                                      \
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);          \
    _dataVec    = IVP_SELNX16I(IVP_MOVNX16_FROM8(gdata), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
  }
  return(_dataVec);
}
#endif

#ifdef IVP_GATHERANX8U
#undef IVP_GATHERANX8U
static inline xb_vecNx16U IVP_GATHERANX8U(const unsigned char * _base, xb_vecNx16U _offsets)
{
  const unsigned char *_basePtr = _base;
  xb_vecNx16U _offsetsVec       = _offsets;
  xb_vecNx16U _dataVec          = (xb_vecNx16U) 0;
  int _i;
  for (_i = 0; _i < 32; _i++)
  {
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);
    xb_int8U gdata        = IVP_LS2NX8U_X(_basePtr, offset);
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELNX16UI(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROM8U(gdata)), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);
  }
  return(_dataVec);
}
#endif

#ifndef IVP_GATHERD2NX8_L
#define IVP_GATHERD2NX8_L(_gsr)  IVP_SEL2NX8I((xb_vec2Nx8) 0, IVP_MOV2NX8_FROMNX16(_gsr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0)
#endif

#ifndef IVP_GATHERD2NX8_H
#define IVP_GATHERD2NX8_H(_vec, _gsr)  do { xb_vec2Nx8 tmp = IVP_SEL2NX8I((xb_vec2Nx8) 0, IVP_MOV2NX8_FROMNX16(_gsr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0); \
                                            _vec = IVP_SEL2NX8I(tmp, _vec, IVP_SELI_EXTRACT_LO_HALVES);                                                  \
} while (0)
#endif

#ifndef IVP_GATHERD2NX8U_H
#define IVP_GATHERD2NX8U_H(_vec, _gsr)  do { xb_vec2Nx8U tmp = IVP_SEL2NX8UI((xb_vec2Nx8U) 0, IVP_MOV2NX8U_FROMNX16(_gsr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0); \
                                             _vec = IVP_SEL2NX8UI(tmp, _vec, IVP_SELI_EXTRACT_LO_HALVES); } while (0)
#endif

#ifndef IVP_GATHERD2NX8U_L
#define IVP_GATHERD2NX8U_L(_gsr)  IVP_SEL2NX8UI((xb_vec2Nx8) 0, IVP_MOV2NX8_FROMNX16(_gsr), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0)
#endif

#ifdef IVP_SCATTERNX8U
#undef IVP_SCATTERNX8U
#define IVP_SCATTERNX8U(_dataIn, _base, _offsets)  do {                                   \
    xb_vecNx16U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SSNX8U_X(_dataVec, _basePtr, offset);                                           \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16UI(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERNX8UT
#undef IVP_SCATTERNX8UT
#define IVP_SCATTERNX8UT(_dataIn, _base, _offsets, _vbr)  do {                            \
    xb_vecNx16U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    xb_vecNx16 _condsVec    = IVP_MOVNX16T((xb_vecNx16) 1, (xb_vecNx16) 0, _vbr);         \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      short cond            = IVP_MOVAV16(_condsVec);                                     \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      if (cond) {                                                                         \
        IVP_SSNX8U_X(_dataVec, _basePtr, offset); }                                       \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16UI(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);       \
      _condsVec   = IVP_SELNX16I(_condsVec, _condsVec, IVP_SELI_16B_ROTATE_RIGHT_1);      \
    }                                                                                     \
} while (0)
#endif


#ifdef IVP_SCATTER2NX8_L
#undef IVP_SCATTER2NX8_L
#define IVP_SCATTER2NX8_L(_dataIn, _base, _offsets)  do {                                 \
    xb_vec2Nx8 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    signed char *_basePtr   = _base;                                                      \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SS2NX8_X(_dataVec, _basePtr, offset);                                           \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);         \
    }                                                                                     \
} while (0)
#endif


#ifdef IVP_SCATTER2NX8T_L
#undef IVP_SCATTER2NX8T_L
#define IVP_SCATTER2NX8T_L(_dataIn, _base, _offsets, _vbr)  do {                          \
    xb_vec2Nx8 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    signed char *_basePtr   = _base;                                                      \
    xb_vec2Nx8 _condsVec    = IVP_MOV2NX8T((xb_vec2Nx8) 1, (xb_vec2Nx8) 0, _vbr);         \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      signed char cond      = IVP_MOVAV8(_condsVec);                                      \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      if (cond) {                                                                         \
        IVP_SS2NX8_X(_dataVec, _basePtr, offset); }                                       \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);         \
      _condsVec   = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8U_L
#undef IVP_SCATTER2NX8U_L
#define IVP_SCATTER2NX8U_L(_dataIn, _base, _offsets)  do {                                \
    xb_vec2Nx8U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SS2NX8U_X(_dataVec, _basePtr, offset);                                          \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);        \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8UT_L
#undef IVP_SCATTER2NX8UT_L
#define IVP_SCATTER2NX8UT_L(_dataIn, _base, _offsets, _vbr)  do {                         \
    xb_vec2Nx8U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    xb_vec2Nx8 _condsVec    = IVP_MOV2NX8T((xb_vec2Nx8) 1, (xb_vec2Nx8) 0, _vbr);         \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      signed char cond      = IVP_MOVAV8(_condsVec);                                      \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      if (cond) {                                                                         \
        IVP_SS2NX8U_X(_dataVec, _basePtr, offset); }                                      \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);        \
      _condsVec   = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8_H
#undef IVP_SCATTER2NX8_H
#define IVP_SCATTER2NX8_H(_dataIn, _base, _offsets)  do {                                 \
    xb_vec2Nx8 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    signed char *_basePtr   = _base;                                                      \
    _dataVec = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_EXTRACT_HI_HALVES);           \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SS2NX8_X(_dataVec, _basePtr, offset);                                           \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);         \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8U_H
#undef IVP_SCATTER2NX8U_H
#define IVP_SCATTER2NX8U_H(_dataIn, _base, _offsets)  do {                                \
    xb_vec2Nx8U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    _dataVec = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_EXTRACT_HI_HALVES);          \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SS2NX8U_X(_dataVec, _basePtr, offset);                                          \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);        \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8T_H
#undef IVP_SCATTER2NX8T_H
#define IVP_SCATTER2NX8T_H(_dataIn, _base, _offsets, _vbr)  do {                          \
    xb_vec2Nx8 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    signed char *_basePtr   = _base;                                                      \
    xb_vec2Nx8 _condsVec    = IVP_MOV2NX8T((xb_vec2Nx8) 1, (xb_vec2Nx8) 0, _vbr);         \
    _dataVec  = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_EXTRACT_HI_HALVES);          \
    _condsVec = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_EXTRACT_HI_HALVES);        \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      signed char cond      = IVP_MOVAV8(_condsVec);                                      \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      if (cond) {                                                                         \
        IVP_SS2NX8_X(_dataVec, _basePtr, offset); }                                       \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8I(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);         \
      _condsVec   = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTER2NX8UT_H
#undef IVP_SCATTER2NX8UT_H
#define IVP_SCATTER2NX8UT_H(_dataIn, _base, _offsets, _vbr)  do {                         \
    xb_vec2Nx8U _dataVec    = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    unsigned char *_basePtr = _base;                                                      \
    xb_vec2Nx8 _condsVec    = IVP_MOV2NX8T((xb_vec2Nx8) 1, (xb_vec2Nx8) 0, _vbr);         \
    _dataVec  = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_EXTRACT_HI_HALVES);         \
    _condsVec = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_EXTRACT_HI_HALVES);        \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      signed char cond      = IVP_MOVAV8(_condsVec);                                      \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      if (cond) {                                                                         \
        IVP_SS2NX8U_X(_dataVec, _basePtr, offset); }                                      \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SEL2NX8UI(_dataVec, _dataVec, IVP_SELI_8B_ROTATE_RIGHT_1);        \
      _condsVec   = IVP_SEL2NX8I(_condsVec, _condsVec, IVP_SELI_8B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERNX16
#undef IVP_SCATTERNX16
#define IVP_SCATTERNX16(_dataIn, _base, _offsets)  do {                                   \
    xb_vecNx16 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    short *_basePtr         = _base;                                                      \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SSNX16_X(_dataVec, _basePtr, offset);                                           \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16I(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);        \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERNX16U
#undef IVP_SCATTERNX16U
#define IVP_SCATTERNX16U(_dataIn, _base, _offsets)  do {                                  \
    xb_vecNx16U _dataVec     = _dataIn;                                                   \
    xb_vecNx16U _offsetsVec  = _offsets;                                                  \
    unsigned short *_basePtr = _base;                                                     \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      IVP_SSNX16U_X(_dataVec, _basePtr, offset);                                          \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16UI(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);       \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERNX16T
#undef IVP_SCATTERNX16T
#define IVP_SCATTERNX16T(_dataIn, _base, _offsets, _vbr)  do {                            \
    xb_vecNx16 _dataVec     = _dataIn;                                                    \
    xb_vecNx16U _offsetsVec = _offsets;                                                   \
    short *_basePtr         = (short *) _base;                                            \
    xb_vecNx16 _condsVec    = IVP_MOVNX16T((xb_vecNx16) 1, (xb_vecNx16) 0, _vbr);         \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      short cond            = IVP_MOVAV16(_condsVec);                                     \
      if (cond) {                                                                         \
        IVP_SSNX16_X(_dataVec, _basePtr, offset); }                                       \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16I(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);        \
      _condsVec   = IVP_SELNX16I(_condsVec, _condsVec, IVP_SELI_16B_ROTATE_RIGHT_1);      \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERNX16UT
#undef IVP_SCATTERNX16UT
#define IVP_SCATTERNX16UT(_dataIn, _base, _offsets, _vbr)  do {                           \
    xb_vecNx16U _dataVec     = _dataIn;                                                   \
    xb_vecNx16U _offsetsVec  = _offsets;                                                  \
    unsigned short *_basePtr = _base;                                                     \
    xb_vecNx16 _condsVec     = IVP_MOVNX16T((xb_vecNx16) 1, (xb_vecNx16) 0, _vbr);        \
    int _i;                                                                               \
    for (_i = 0; _i < 32; _i++) {                                                         \
      unsigned short offset = IVP_MOVAVU16(_offsetsVec);                                  \
      short cond            = IVP_MOVAV16(_condsVec);                                     \
      if (cond) {                                                                         \
        IVP_SSNX16U_X(_dataVec, _basePtr, offset); }                                      \
      _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELNX16UI(_dataVec, _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);       \
      _condsVec   = IVP_SELNX16I(_condsVec, _condsVec, IVP_SELI_16B_ROTATE_RIGHT_1);      \
    }                                                                                     \
} while (0)
#endif

#ifdef IVP_SCATTERN_2X32
#undef IVP_SCATTERN_2X32
#define IVP_SCATTERN_2X32(_dataIn, _base, _offsets)  do {                                   \
    xb_vecN_2x32v _dataVec     = _dataIn;                                                   \
    xb_vecN_2x32Uv _offsetsVec = _offsets;                                                  \
    int *_basePtr              = _base;                                                     \
    int _i;                                                                                 \
    for (_i = 0; _i < 16; _i++) {                                                           \
      unsigned int offset = IVP_MOVAV32(_offsetsVec);                                       \
      IVP_SSN_2X32_X(_dataVec, _basePtr, offset);                                           \
      _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELN_2X32I(_dataVec, _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);        \
    }                                                                                       \
} while (0)
#endif

#ifdef IVP_SCATTERN_2X32T
#undef IVP_SCATTERN_2X32T
#define IVP_SCATTERN_2X32T(_dataIn, _base, _offsets, _vbr)  do {                             \
    xb_vecN_2x32v _dataVec     = _dataIn;                                                    \
    xb_vecN_2x32Uv _offsetsVec = _offsets;                                                   \
    int *_basePtr              = _base;                                                      \
    xb_vecN_2x32v _condsVec    = IVP_MOVN_2X32T((xb_vecN_2x32v) 1, (xb_vecN_2x32v) 0, _vbr); \
    int _i;                                                                                  \
    for (_i = 0; _i < 16; _i++) {                                                            \
      int cond            = IVP_MOVAV32(_condsVec);                                          \
      unsigned int offset = IVP_MOVAV32(_offsetsVec);                                        \
      if (cond) {                                                                            \
        IVP_SSN_2X32_X(_dataVec, _basePtr, offset); }                                        \
      _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1);  \
      _dataVec    = IVP_SELN_2X32I(_dataVec, _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);         \
      _condsVec   = IVP_SELN_2X32I(_condsVec, _condsVec, IVP_SELI_32B_ROTATE_RIGHT_1);       \
    }                                                                                        \
} while (0)
#endif

#ifdef IVP_SCATTERN_2X32U
#undef IVP_SCATTERN_2X32U
#define IVP_SCATTERN_2X32U(_dataIn, _base, _offsets)  do {                                  \
    xb_vecN_2x32Uv _dataVec    = _dataIn;                                                   \
    xb_vecN_2x32Uv _offsetsVec = _offsets;                                                  \
    unsigned int *_basePtr     = _base;                                                     \
    int _i;                                                                                 \
    for (_i = 0; _i < 16; _i++) {                                                           \
      unsigned int offset = IVP_MOVAV32(_offsetsVec);                                       \
      IVP_SSN_2X32U_X(_dataVec, _basePtr, offset);                                          \
      _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1); \
      _dataVec    = IVP_SELN_2X32UI(_dataVec, _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);       \
    }                                                                                       \
} while (0)
#endif

#ifdef IVP_SCATTERN_2X32UT
#undef IVP_SCATTERN_2X32UT
#define IVP_SCATTERN_2X32UT(_dataIn, _base, _offsets, _vbr)  do {                            \
    xb_vecN_2x32Uv _dataVec    = _dataIn;                                                    \
    xb_vecN_2x32Uv _offsetsVec = _offsets;                                                   \
    unsigned int *_basePtr     = _base;                                                      \
    xb_vecN_2x32v _condsVec    = IVP_MOVN_2X32T((xb_vecN_2x32v) 1, (xb_vecN_2x32v) 0, _vbr); \
    int _i;                                                                                  \
    for (_i = 0; _i < 16; _i++) {                                                            \
      int cond            = IVP_MOVAV32(_condsVec);                                          \
      unsigned int offset = IVP_MOVAV32(_offsetsVec);                                        \
      if (cond) {                                                                            \
        IVP_SSN_2X32U_X(_dataVec, _basePtr, offset); }                                       \
      _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1);  \
      _dataVec    = IVP_SELN_2X32UI(_dataVec, _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);        \
      _condsVec   = IVP_SELN_2X32I(_condsVec, _condsVec, IVP_SELI_32B_ROTATE_RIGHT_1);       \
    }                                                                                        \
} while (0)
#endif

#ifdef IVP_GATHERANX16U
#undef IVP_GATHERANX16U
static inline xb_vecNx16 IVP_GATHERANX16U(const uint16_t *_base, xb_vecNx16U _offsets)
{
  const unsigned short *_basePtr = _base;
  xb_vecNx16U _offsetsVec        = _offsets;
  xb_vecNx16U _dataVec           = (xb_vecNx16U) 0;
  int _i;
  for (_i = 0; _i < 32; _i++)
  {
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);
    xb_int16U gdata       = IVP_LSNX16U_X(_basePtr, offset);
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELNX16UI(IVP_MOVNX16U_FROM16U(gdata), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);
  }
  return(IVP_MOVNX16_FROMNX16U(_dataVec));
}
#endif

#ifdef IVP_GATHERANX16
#undef IVP_GATHERANX16
static inline xb_vecNx16 IVP_GATHERANX16(const int16_t *_base, xb_vecNx16U _offsets)
{
  const short *_basePtr   = _base;
  xb_vecNx16U _offsetsVec = _offsets;
  xb_vecNx16 _dataVec     = (xb_vecNx16) 0;
  int _i;
  for (_i = 0; _i < 32; _i++)
  {
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);
    xb_int16 gdata        = IVP_LSNX16_X(_basePtr, offset);
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELNX16I(IVP_MOVNX16_FROM16(gdata), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);
  }
  return(_dataVec);
}
#endif

#ifdef IVP_GATHERANX16T
#undef IVP_GATHERANX16T
static inline xb_vecNx16 IVP_GATHERANX16T(const int16_t *_base, xb_vecNx16U _offsets, vboolN _vbr)
{
  const short *_basePtr   = _base;
  xb_vecNx16U _offsetsVec = _offsets;
  vboolN _boolVec         = _vbr;
  xb_vecNx16 _dataVec     = (xb_vecNx16) 0;
  int _i;
  for (_i = 0; _i < 32; _i++)
  {
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);
    xb_int16 gdata        = IVP_LSNX16_X(_basePtr, offset);
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELNX16I(IVP_MOVNX16_FROM16(gdata), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);
  }
  return(IVP_MOVNX16T(_dataVec, (xb_vecNx16) 0, _boolVec));
}
#endif

#ifdef IVP_GATHERANX16UT
#undef IVP_GATHERANX16UT
static inline xb_vecNx16 IVP_GATHERANX16UT(const uint16_t *_base, xb_vecNx16U _offsets, vboolN _vbr)
{
  const unsigned short *_basePtr = _base;
  xb_vecNx16U _offsetsVec        = _offsets;
  vboolN _boolVec                = _vbr;
  xb_vecNx16U _dataVec           = (xb_vecNx16U) 0;
  int _i;
  for (_i = 0; _i < 32; _i++)
  {
    unsigned short offset = IVP_MOVAVU16(_offsetsVec);
    xb_int16U gdata       = IVP_LSNX16U_X(_basePtr, offset);
    _offsetsVec = IVP_SELNX16UI(_offsetsVec, _offsetsVec, IVP_SELI_16B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELNX16UI(IVP_MOVNX16U_FROM16U(gdata), _dataVec, IVP_SELI_16B_ROTATE_RIGHT_1);
  }
  return(IVP_MOVNX16_FROMNX16U(IVP_MOVNX16UT(_dataVec, (xb_vecNx16U) 0, _boolVec)));
}
#endif

#ifdef IVP_GATHERAN_2X32
#undef IVP_GATHERAN_2X32
static inline xb_vecNx16 IVP_GATHERAN_2X32(const int32_t *_base, xb_vecN_2x32Uv _offsets)
{
  const int *_basePtr        = _base;
  xb_vecN_2x32Uv _offsetsVec = _offsets;
  xb_vecN_2x32v _dataVec     = (xb_vecN_2x32v) 0;
  int _i;
  for (_i = 0; _i < 16; _i++)
  {
    unsigned int offset = IVP_MOVAV32(_offsetsVec);
    xb_int32v gdata     = IVP_LSN_2X32_X(_basePtr, offset);
    _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELN_2X32I(IVP_MOVN_2X32_FROM32(gdata), _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);
  }
  return(IVP_MOVNX16_FROMN_2X32(_dataVec));
}
#endif

#ifdef IVP_GATHERAN_2X32T
#undef IVP_GATHERAN_2X32T
static inline xb_vecNx16  IVP_GATHERAN_2X32T(const int32_t *_base, xb_vecN_2x32Uv _offsets, vboolN_2 _vbr)
{
  const int *_basePtr        = _base;
  xb_vecN_2x32Uv _offsetsVec = _offsets;
  vboolN_2 _boolVec          = _vbr;
  xb_vecN_2x32v _dataVec     = (xb_vecN_2x32v) 0;
  int _i;
  for (_i = 0; _i < 16; _i++)
  {
    unsigned int offset = IVP_MOVAV32(_offsetsVec);
    xb_int32v gdata     = IVP_LSN_2X32_X(_basePtr, offset);
    _offsetsVec = IVP_SELN_2X32UI(_offsetsVec, _offsetsVec, IVP_SELI_32B_ROTATE_RIGHT_1);
    _dataVec    = IVP_SELN_2X32I(IVP_MOVN_2X32_FROM32(gdata), _dataVec, IVP_SELI_32B_ROTATE_RIGHT_1);
  }
  return(IVP_MOVNX16_FROMN_2X32(IVP_MOVN_2X32T(_dataVec, (xb_vecN_2x32v) 0, _boolVec)));
}
#endif

#ifdef IVP_GATHERNX8UT_V
#undef IVP_GATHERNX8UT_V
#define IVP_GATHERNX8UT_V(pdst, offs, mask, dly)  IVP_MOVNX16T(IVP_GATHERNX8U_V((pdst), (offs), (dly)), 0, mask)
#endif
#endif // XCHAL_HAVE_SUPERGATHER == 0

////////// protos extension

// 32-way wide vector (48-bit) element high 16-bits output to narrow (16-bit) output vector register
#ifndef IVP_PACKHNX48
#   define IVP_PACKHNX48(vec)  IVP_PACKVRNR2NX24_1(IVP_MOV2NX24_FROMNX48(vec), 8)
#endif

// reinterpret 64 8-bit elements as 16 32-bit elements
#ifndef IVP_MOVN_2X32_FROM2NX8
#   define IVP_MOVN_2X32_FROM2NX8(vec)  IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(vec))
#endif

// reinterpret 16 32-bit elements as 64 8-bit elements
#ifndef IVP_MOV2NX8_FROMN_2X32
#   define IVP_MOV2NX8_FROMN_2X32(vec)  IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(vec))
#endif

#ifndef IVP_SELN_2X32I
#   define IVP_SELN_2X32I(a, b, i)  IVP_MOVN_2X32_FROMNX16(IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32(a), IVP_MOVNX16_FROMN_2X32(b), i))
#endif

// 0 to 63 sequence xb_vec2Nx8U vector
#ifndef IVP_SEQ2NX8U
#   define IVP_SEQ2NX8U()  IVP_MOV2NX8U_FROMNX16(IVP_ADDNX16U(256, IVP_MULNX16UPACKL(514, IVP_SEQNX16())))
#endif

// 64-way 8-bit zero
#ifndef IVP_ZERO2NX8U
#   define IVP_ZERO2NX8U()  IVP_MOV2NX8U_FROMNX16(IVP_ZERONX16())
#endif

// 16-way 32-bit zero
#ifndef IVP_ZERON_2X32U
#   define IVP_ZERON_2X32U()  IVP_MOVN_2X32U_FROMNX16(IVP_ZERONX16())
#endif

// 64-way 24-bit zero
#ifndef IVP_ZERO2NX24
#   define IVP_ZERO2NX24()  IVP_MOV2NX24_FROMNX48(IVP_ZERONX48())
#endif

// 32-way 48-bit zero
#ifndef IVP_ZERONX48
#   if XCHAL_HAVE_VISION
#       define IVP_ZERONX48()  (IVP_CVT48UNX32L(IVP_ZERON_2X32U()))
#   else
#       define IVP_ZERONX48()  (IVP_MOVWVL(IVP_ZERONX16()))
#   endif
#endif

////////// compatibility between IVPEP - VP5
#if XCHAL_HAVE_VISION

typedef xb_vecNx16  vsaN;

#   define IVP_MOVWVL(a)                  IVP_CVT48UNX32L(a)
#   define IVP_MOVV2WHH(a)                IVP_MOVNX16_FROMN_2X32(IVP_CVT32S2NX24HH(IVP_MOV2NX24_FROMNX48(a)))
#   define IVP_MOVV2WHL(a)                IVP_MOVNX16_FROMN_2X32(IVP_CVT32S2NX24HL(IVP_MOV2NX24_FROMNX48(a)))
#   define IVP_MOVV2WLH(a)                IVP_MOVNX16_FROMN_2X32(IVP_CVT32S2NX24LH(IVP_MOV2NX24_FROMNX48(a)))
#   define IVP_MOVV2WLL(a)                IVP_MOVNX16_FROMN_2X32(IVP_CVT32S2NX24LL(IVP_MOV2NX24_FROMNX48(a)))
#   define IVP_MOVSVWH(a)                 IVP_MOVNX16_FROMN_2X32(IVP_CVT32SNX48H(a))
#   define IVP_MOVSVWL(a)                 IVP_MOVNX16_FROMN_2X32(IVP_CVT32SNX48L(a))
#   define IVP_MOVVWHH(a)                 IVP_MOVNX16_FROM2NX8(IVP_CVT64SNX48HH(a))
#   define IVP_MOVVWHL(a)                 IVP_MOVNX16_FROM2NX8(IVP_CVT64SNX48HL(a))
#   define IVP_MOVVWLH(a)                 IVP_MOVNX16_FROM2NX8(IVP_CVT64SNX48LH(a))
#   define IVP_MOVVWLL(a)                 IVP_MOVNX16_FROM2NX8(IVP_CVT64SNX48LL(a))
#   define IVP_MOVV2WL(a)                 IVP_CVT16U2NX24L(IVP_MOV2NX24_FROMNX48(a))
#   define IVP_MOVV2WH(a)                 IVP_CVT16U2NX24H(IVP_MOV2NX24_FROMNX48(a))
#   define IVP_MOVVWL(a)                  IVP_MOVNX16_FROMN_2X32(IVP_CVT32UNX48L(a))
#   define IVP_MOVVWH(a)                  IVP_MOVNX16_FROMN_2X32(IVP_CVT32UNX48H(a))
#   define IVP_MOVSV2WL(a)                IVP_CVT16S2NX24L(IVP_MOV2NX24_FROMNX48(a))
#   define IVP_MOVSV2WH(a)                IVP_CVT16S2NX24H(IVP_MOV2NX24_FROMNX48(a))
#   define IVP_MOV2W2VL(a, b)             IVP_MOVNX48_FROM2NX24(IVP_CVT24UNX32L(IVP_MOVN_2X32_FROMNX16(a), IVP_MOVN_2X32_FROMNX16(b)))
#   define IVP_MOVSWV(a, b)               IVP_CVT48SNX32(IVP_MOVN_2X32_FROMNX16(a), IVP_MOVN_2X32_FROMNX16(b))
#   define IVP_MOVS2WV(a, b)              IVP_MOVNX48_FROM2NX24(IVP_CVT24S2NX16(a, b))
#   define IVP_MOVWV(a, b)                IVP_CVT48UNX32(IVP_MOVN_2X32_FROMNX16(a), IVP_MOVN_2X32_FROMNX16(b))

#   define IVP_MOVVVS(a)                  (a)
#   define IVP_MOVVSA32(a)                IVP_MOVVA16(a)
#   define IVP_MOVVSV(vr, sa)             (vr) // sa is always zero in XI, if not zero -> use IVP_MOVVSELNX16
#   define IVP_MOVVSELNX16(vr, sa)        IVP_SRLINX16(vr, sa)
#   define IVP_MOVVSVADDNX16(a, b, c, d)  { a = c; c = IVP_ADDNX16(c, b); } // d is always zero in XI
#   define IVP_MOVPVSV(a, b, c, d)        { xb_vec2Nx8 t = IVP_SRLI2NX8(c, d); a = IVP_UNPKS2NX8_1(t); b = IVP_UNPKS2NX8_0(t); }

#undef IVP_LSNX8U_XP
#undef IVP_LSNX8U_IP
#undef IVP_LSNX8U_X
#undef IVP_LSNX8U_I
#   define IVP_LSNX8U_XP(a, b, c)  do { xb_int8U tmp; IVP_LS2NX8U_XP(tmp, b, c); a = IVP_MOVNX16_FROM8U(tmp); } while (0)
#   define IVP_LSNX8U_IP(a, b, c)  do { xb_int8U tmp; IVP_LS2NX8U_IP(tmp, b, c); a = IVP_MOVNX16_FROM8U(tmp); } while (0)
#   define IVP_LSNX8U_X(b, c)      IVP_MOVNX16_FROM8U(IVP_LS2NX8U_X(b, c))
#   define IVP_LSNX8U_I(b, c)      IVP_MOVNX16_FROM8U(IVP_LS2NX8U_I(b, c))

#   define IVP_PACKLNX48_L(a)      IVP_CVT32UNX48L(a)
#   define IVP_PACKLNX48_H(a)      IVP_CVT32UNX48H(a)

#   define IVP_SA2NX8UPOS_FP    IVP_SAPOS2NX8U_FP
#   define IVP_SAN_2X32POS_FP   IVP_SAPOSN_2X32_FP
#   define IVP_SANX16POS_FP     IVP_SAPOSNX16_FP
#   define IVP_SANX16UPOS_FP    IVP_SAPOSNX16U_FP
#   define IVP_SANX8UPOS_FP     IVP_SAPOSNX8U_FP
#   define IVP_SAV2NX8POS_FP    IVP_SAPOS2NX8_FP
#   define IVP_SAV2NX8UPOS_FP   IVP_SAPOS2NX8U_FP
#   define IVP_SAVN_2X32POS_FP  IVP_SAPOSN_2X32_FP
#   define IVP_SAVNX16POS_FP    IVP_SAPOSNX16_FP
#   define IVP_SAVNX16UPOS_FP   IVP_SAPOSNX16U_FP
#   define IVP_SAVNX8UPOS_FP    IVP_SAPOSNX8U_FP
#   define IVP_LAVNX8U_PP       IVP_LANX8U_PP
#   define IVP_LAVNX16_PP       IVP_LANX16_PP

#   define IVP_RADDURNX16(b)           ((int) IVP_RADDUNX16(b))
#   define IVP_RADDRNX16(b)            ((int) IVP_RADDNX16(b))
#   define IVP_ADDSNX16F(a, b, c, d)   IVP_ADDSNX16T(a, b, c, IVP_NOTBN(d))
#   define IVP_ADDNX16F(a, b, c, d)    IVP_ADDNX16T(a, b, c, IVP_NOTBN(d))
#   define IVP_SUBNX16F(a, b, c, d)    IVP_SUBNX16T(a, b, c, IVP_NOTBN(d))
#   define IVP_NEGNX16F(a, b, c)       IVP_NEGNX16T(a, b, IVP_NOTBN(c))
#   define IVP_NEGSNX16F(a, b, c)      IVP_NEGSNX16T(a, b, IVP_NOTBN(c))
#   define IVP_RMINNX16F(b, c)         IVP_RMINNX16T(b, IVP_NOTBN(c))
#   define IVP_MINUNX16F(a, b, c, d)   IVP_MINUNX16T(a, b, c, IVP_NOTBN(d))
#   define IVP_SVNX8UF_XP(a, b, c, d)  IVP_SVNX8UT_XP(a, b, c, IVP_NOTBN(d))
#   define IVP_SVNX8UF_I(a, b, c, d)   IVP_SVNX8UT_I(a, b, c, IVP_NOTBN(d))
#   define IVP_SVNX16F_XP(a, b, c, d)  IVP_SVNX16T_XP(a, b, c, IVP_NOTBN(d))
#   define IVP_SVNX16F_I(a, b, c, d)   IVP_SVNX16T_I(a, b, c, IVP_NOTBN(d))
#endif

#if XCHAL_HAVE_VISION
#   define IVP__LSNX16_XP(a, b, c)  do { xb_int16 tmp; IVP_LSNX16_XP(tmp, b, c); a = IVP_MOVNX16_FROM16(tmp); } while (0)
#else
#   define IVP__LSNX16_XP  IVP_LSNX16_XP
#endif

#if XCHAL_HAVE_VISION
#   define IVP__LSNX16_IP(a, b, c)  do { xb_int16 tmp; IVP_LSNX16_IP(tmp, b, c); a = IVP_MOVNX16_FROM16(tmp); } while (0)
#else
#   define IVP__LSNX16_IP  IVP_LSNX16_IP
#endif

#if XCHAL_HAVE_VISION
#   define IVP__DSELNX16_2X16(a, b, c, d, e, f)  { \
    xb_vecNx16 _v0, _v1;                           \
    _v0 = d;                                       \
    _v1 = c;                                       \
    a   = IVP_SELNX16(_v1, _v0, e);                \
    b   = IVP_SELNX16(_v1, _v0, f);                \
}
#else
#   define IVP__DSELNX16_2X16  IVP_DSELNX16
#endif

#if XCHAL_HAVE_VISION
#   define IVP__SEL2NX8_2X16(b, c, d, e)  IVP_SEL2NX8(b, c, IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(d), IVP_MOV2NX8_FROMNX16(e), IVP_SELI_8B_INTERLEAVE_1_EVEN))
#else
#   define IVP__SEL2NX8_2X16  IVP_SEL2NX8
#endif

////////// compatibility for RF-2014.0 IVP-EP cores

#ifndef IVP_SVN_2X32_IP
#define IVP_SVN_2X32_IP(a, b, c)                     \
  do {                                               \
    xb_vecNx16 *bb = (xb_vecNx16 *) b;               \
    IVP_SVNX16_IP(IVP_MOVNX16_FROMN_2X32(a), bb, c); \
    b = (xb_vecN_2x32v *) bb;                        \
  } while (0)
#endif

#ifndef IVP_SVN_2X32_XP
#define IVP_SVN_2X32_XP(a, b, c)                     \
  do {                                               \
    xb_vecNx16 *bb = (xb_vecNx16 *) b;               \
    IVP_SVNX16_XP(IVP_MOVNX16_FROMN_2X32(a), bb, c); \
    b = (xb_vecN_2x32v *) bb;                        \
  } while (0)
#endif

#ifndef IVP_LVN_2X32_IP
#define IVP_LVN_2X32_IP(a, b, c)             \
  do {                                       \
    xb_vecNx16 *bb = (xb_vecNx16 *) b;       \
    xb_vecNx16 aa; IVP_LVNX16_IP(aa, bb, c); \
    a = IVP_MOVN_2X32_FROMNX16(aa);          \
    b = (xb_vecN_2x32v *) bb;                \
  } while (0)
#endif

#ifndef IVP_LVN_2X32_XP
#define IVP_LVN_2X32_XP(a, b, c)             \
  do {                                       \
    xb_vecNx16 *bb = (xb_vecNx16 *) b;       \
    xb_vecNx16 aa; IVP_LVNX16_XP(aa, bb, c); \
    a = IVP_MOVN_2X32_FROMNX16(aa);          \
    b = (xb_vecN_2x32v *) bb;                \
  } while (0)
#endif

////////// select/shuffle indexes
#if XCHAL_HAVE_VISION
#define XAI_DSEL_16B_ROTATE_LEFT(n)   IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16((0x4000 - 2 * (((n) << 8) + (n)))))
#define XAI_DSEL_16B_ROTATE_RIGHT(n)  IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16((0x3F00 + 2 * (((n) << 8) + (n)))))

#define XAI_DSEL_16B_ROTATE_RIGHT_2_1  IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16(2 * (1 + ((1 + 1) << 8))))
#define XAI_DSEL_16B_ROTATE_RIGHT_4_3  IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16(2 * (3 + ((3 + 1) << 8))))
#define XAI_DSEL_32B_ROTATE_RIGHT_2_1  IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16(4 * (1 + ((1 + 1) << 8))))
#define XAI_DSEL_32B_ROTATE_RIGHT_4_3  IVP_AVGU2NX8(IVP_SEQ2NX8(), IVP_MOV2NX8_FROMNX16(4 * (3 + ((3 + 1) << 8))))
#endif

#define OFFSET_PTR_NX8(ptr, nrows, stride, in_row_offset)    ((xb_vecNx8 *)     ((int8_t *)  (ptr) + (in_row_offset) + (nrows) * (stride)))
#define OFFSET_PTR_NX8U(ptr, nrows, stride, in_row_offset)   ((xb_vecNx8U *)    ((uint8_t *) (ptr) + (in_row_offset) + (nrows) * (stride)))
#define OFFSET_PTR_2NX8(ptr, nrows, stride, in_row_offset)   ((xb_vec2Nx8 *)    ((int8_t *)  (ptr) + (in_row_offset) + (nrows) * (stride)))
#define OFFSET_PTR_2NX8U(ptr, nrows, stride, in_row_offset)  ((xb_vec2Nx8U *) ((uint8_t *) (ptr) + (in_row_offset) + (nrows) * (stride)))
#define OFFSET_PTR_NX16(ptr, nrows, stride, in_row_offset)   ((xb_vecNx16 *)    ((int16_t *) (ptr) + (in_row_offset) + (nrows) * (stride)))
#define OFFSET_PTR_NX16U(ptr, nrows, stride, in_row_offset)  ((xb_vecNx16U *) ((uint16_t *) (ptr) + (in_row_offset) + (nrows) * (stride)))
#endif
