/*
 * Copyright (c) 2025 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
 * These coded instructions, statements, and computer programs are the
 * copyrighted works and confidential proprietary information of
 * Cadence Design Systems Inc.  They may be adapted and modified by bona fide
 * purchasers for internal use, but neither the original nor any adapted
 * or modified version may be disclosed or distributed to third parties
 * in any manner, medium, or form, in whole or in part, without the prior
 * written consent of Cadence Design Systems Inc.  This software and its
 * derivatives are to be executed solely on products incorporating a Cadence
 * Design Systems processor.
 */

#include "xai_cnn_common.h"

/**************************** xaiCast3DScalar_I64 *******************************/
/* Description  :  Data casting scalar implementation for the case when         */
/*                 input or output data Type are U64 or S64                     */
/* Inputs       : inTile                                                        */
/* Outputs      : void                                                          */
/* InOuts       : outTile                                                       */
/********************************************************************************/

void xaiCast3DScalar_I64(const xai_pTile3D inTile,
                         xai_pTile3D outTile)
{
  /* Get Tile Parameters */
  const int32_t dim1Size  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size  = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inPitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inPitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  uint8_t *pIn_8bU   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pIn_8b     = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint16_t *pIn_16bU = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pIn_16b   = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint32_t *pIn_32bU = (uint32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pIn_32b   = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint64_t *pIn_64bU = (uint64_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pIn_64b   = (int64_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1))
  xb_f16 *pIn_f16b = (xb_f16 *) XAI_TILE3D_GET_DATA_PTR(inTile);
#endif
  float *pIn_f32b = (float *) XAI_TILE3D_GET_DATA_PTR(inTile);

  uint8_t *pout_8bU   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pout_8b     = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint16_t *pout_16bU = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int16_t *pout_16b   = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint32_t *pout_32bU = (uint32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t *pout_32b   = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint64_t *pout_64bU = (uint64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t *pout_64b   = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1))
  xb_f16 *pout_f16b = (xb_f16 *) XAI_TILE3D_GET_DATA_PTR(outTile);
#endif
  float *pout_f32b = (float *) XAI_TILE3D_GET_DATA_PTR(outTile);

  int32_t x, y, z;

  for (z = 0; z < dim3Size; z++) /* along 3rd dimension */
  {
    for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
    {
      for (x = 0; x < dim1Size; x++) /* along 1st dimension */
      {
        // Conversions to U64
        if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U8 -> U64
            case XAI_U8:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> U64
            case XAI_S8:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> U64
            case XAI_U16:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> U64
            case XAI_S16:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> U64
            case XAI_U32:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> U64
            case XAI_S32:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U64
            case XAI_S64:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> U64
            case XAI_F32:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1))
            // F16 -> U64
            case XAI_F16:
              pout_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) IVP_CVTF32F16(pIn_f16b[z * inPitch2 + y * inPitch1 + x]);
              break;
#endif
            default:
              break;
          }
        }
        // Conversions to S64
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U8 -> S64
            case XAI_U8:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> S64
            case XAI_S8:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> S64
            case XAI_U16:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> S64
            case XAI_S16:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> S64
            case XAI_U32:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> S64
            case XAI_S32:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S64
            case XAI_U64:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> S64
            case XAI_F32:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1))
            // F16 -> S64
            case XAI_F16:
              pout_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) IVP_CVTF32F16(pIn_f16b[z * inPitch2 + y * inPitch1 + x]);
              break;
#endif
            default:
              break;
          }
        }
        // Conversions to S32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> U32
            case XAI_U64:
              pout_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U32
            case XAI_S64:
              pout_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to S32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> S32
            case XAI_U64:
              pout_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S32
            case XAI_S64:
              pout_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to U16
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> U16
            case XAI_U64:
              pout_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U16
            case XAI_S64:
              pout_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to S16
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> S16
            case XAI_U64:
              pout_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S16
            case XAI_S64:
              pout_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to U8
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> U8
            case XAI_U64:
              pout_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U8
            case XAI_S64:
              pout_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to S8
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> S8
            case XAI_U64:
              pout_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S8
            case XAI_S64:
              pout_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to F32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_F32))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> F32
            case XAI_U64:
              pout_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> F32
            case XAI_S64:
              pout_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            default:
              break;
          }
        }
        // Conversions to F16
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1))
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_F16))
        {
          switch (XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(inTile)))
          {
            // U64 -> F16
            case XAI_U64:
              pout_f16b[z * outPitch2 + y * outPitch1 + x] = IVP_CVTF16F32((float) pIn_64bU[z * inPitch2 + y * inPitch1 + x]);
              break;
            // S64 -> F16
            case XAI_S64:
              pout_f16b[z * outPitch2 + y * outPitch1 + x] = IVP_CVTF16F32((float) pIn_64b[z * inPitch2 + y * inPitch1 + x]);
              break;
            default:
              break;
          }
        }
#endif
      } /* end (x = 0; x < dim1Size; x++) loop */
    }   /* end (y = 0; y < dim2Size; y++) loop */
  }     /* end (z = 0; z < dim3Size; z++) loop */
  return;
}

