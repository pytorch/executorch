/*
 * Copyright (c) 2018 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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


#ifndef SIGNED8BIT
#define SIGNED8BIT     1
#define UNSIGNED8BIT   2
#define SIGNED16BIT    3
#define UNSIGNED16BIT  4
#define SIGNED32BIT    5
#define UNSIGNED32BIT  6
#define FLOAT16BIT     7
#define FLOAT32BIT     8
#endif

#if ELT_LESSTHAN_DATA_TYPE == SIGNED8BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _S8_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_S8
#define MORPH_IDT_SCALAR  int8_t

#elif ELT_LESSTHAN_DATA_TYPE == UNSIGNED8BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _U8_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_U8
#define MORPH_IDT_SCALAR  uint8_t

#elif ELT_LESSTHAN_DATA_TYPE == SIGNED16BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _S16_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_S16
#define MORPH_IDT_SCALAR  int16_t

#elif ELT_LESSTHAN_DATA_TYPE == UNSIGNED16BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _U16_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_U16
#define MORPH_IDT_SCALAR  uint16_t

#elif ELT_LESSTHAN_DATA_TYPE == SIGNED32BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _S32_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_S32
#define MORPH_IDT_SCALAR  int32_t

#elif ELT_LESSTHAN_DATA_TYPE == UNSIGNED32BIT
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _U32_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_U32
#define MORPH_IDT_SCALAR  uint32_t

#elif ELT_LESSTHAN_DATA_TYPE == FLOAT16BIT
#if XCHAL_HAVE_VISION_HP_VFPU == 1
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _F16_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_F16
#define MORPH_IDT_SCALAR  xb_f16
#endif

#elif ELT_LESSTHAN_DATA_TYPE == FLOAT32BIT
#if XCHAL_HAVE_VISION_SP_VFPU == 1
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#define MAKE_NAME(name)  name ## _F32_AV
#define MORPH_IDT_CHECK   XAI_CHECK_TILE3D_F32
#define MORPH_IDT_SCALAR  float
#endif
#endif

#define LESS_THAN(a, b)  a < b


/**************************** xaiEltwiseLessThan3D ***************************************/
/* Description  : auto-vectorizable implementation of Broadcast elementWise LESS         */
/*               operator, Based on MORPH implementation eight variants are              */
/*               generated for S8, U8, S16, U16, S32, U32, F16 and F32 data types        */
/* Inputs       : inTile1, inTile2                                                       */
/* Outputs      : XI Error Code                                                          */
/* InOuts       : outTile                                                                */
/*****************************************************************************************/

XAI_ERR_TYPE MAKE_NAME (xaiEltwiseLessThan3D)(const xai_pTile3D inTile1, const xai_pTile3D inTile2, xai_pTile3D outTile)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_CHECK(inTile1);
    MORPH_IDT_CHECK(inTile2);
    MORPH_IDT_CHECK(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile1);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile2);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile1) == XAI_TILE3D_GET_DATA_ORDER(inTile2),
                    XAI_ERR_BADARG, "\nData Order of InputTile1 = %d and InputTile2 = %d\nData Order of InputTile1 and InputTile2 should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile1), XAI_TILE3D_GET_DATA_ORDER(inTile2));
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile1) == XAI_TILE3D_GET_DATA_ORDER(outTile),
                    XAI_ERR_BADARG, "\nData Order of InputTile = %d and OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile1), XAI_TILE3D_GET_DATA_ORDER(outTile));
    XAI_CHECK_TILE3D_BCAST_DIMENSIONS(inTile1, inTile2, outTile, 1, 1);
  }

  /* Get Tile Parameters */
  const int32_t dim1SizeOut   = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t dim2SizeOut   = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t dim3SizeOut   = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  MORPH_IDT_SCALAR *pInput1 = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile1);
  MORPH_IDT_SCALAR *pInput2 = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile2);
  MORPH_IDT_SCALAR *pOutput = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(outTile);

  MORPH_IDT_SCALAR *__restrict pIn1;
  MORPH_IDT_SCALAR *__restrict pIn2;
  MORPH_IDT_SCALAR *__restrict pOut;

  /* Get Pitch appropriate for elementwise broadcast operations */
  XAI_TILE3D_GET_BCAST123_PITCH(inTile1, inTile2, inTile1Pitch0, inTile2Pitch0, inTile1Pitch1, \
                                inTile2Pitch1, inTile1Pitch2, inTile2Pitch2);

  /* no Broadcast */
  if (inTile1Pitch2 == inTile2Pitch2 && inTile1Pitch2 == outTilePitch2)
  {
    int dimsCount = dim1SizeOut * dim2SizeOut * dim3SizeOut;
    pIn1 = pInput1;
    pIn2 = pInput2;
    pOut = pOutput;

    for (int i = 0; i < dimsCount; i++)
    {
#if (ELT_LESSTHAN_DATA_TYPE == FLOAT16BIT)
      bool temp = LESS_THAN(pIn1[i], pIn2[i]);
      pOut[i] = temp ? 1 : 0;
#else
      pOut[i] = LESS_THAN(pIn1[i], pIn2[i]);
#endif
    }
  }
  else
  {
    /*
       inTile1Pitch0 == 0 : Tile1 Dimension 1 broadcasting
       inTile1Pitch1 == 0 : Tile1 Dimension 2 broadcasting
       inTile1Pitch2 == 0 : Tile1 Dimension 3 broadcasting
       inTile2Pitch0 == 0 : Tile2 Dimension 1 broadcasting
       inTile2Pitch1 == 0 : Tile2 Dimension 2 broadcasting
       inTile2Pitch2 == 0 : Tile2 Dimension 3 broadcasting
     */
    int32_t y, z, idx;

    for (z = 0; z < dim3SizeOut; z++)
    {
      MORPH_IDT_SCALAR* temp1 = pInput1 + z * inTile1Pitch2;
      MORPH_IDT_SCALAR* temp2 = pInput2 + z * inTile2Pitch2;
      MORPH_IDT_SCALAR* temp3 = pOutput + z * outTilePitch2;

      for (y = 0; y < dim2SizeOut; y++)
      {
        pIn1 = (temp1 + y * inTile1Pitch1);
        pIn2 = (temp2 + y * inTile2Pitch1);
        pOut = (temp3 + y * outTilePitch1);

        MORPH_IDT_SCALAR InData1, InData2;
        /* Tile1 Dimension 1 broadcasting */
        if (inTile1Pitch0 == 0)
        {
          InData1 = pIn1[0];
          /* reduced one load from core loop */
          for (idx = 0; idx < dim1SizeOut; idx++)
          {
            InData2 = pIn2[idx];
#if (ELT_LESSTHAN_DATA_TYPE == FLOAT16BIT)
            bool temp = LESS_THAN(InData1, InData2);
            pOut[idx] = temp ? 1 : 0;
#else
            pOut[idx] = LESS_THAN(InData1, InData2);
#endif
          }
        }
        /* Tile2 Dimension 1 broadcasting */
        else if (inTile2Pitch0 == 0)
        {
          InData2 = pIn2[0];
          /* reduced one load from core loop */
          for (idx = 0; idx < dim1SizeOut; idx++)
          {
            InData1 = pIn1[idx];
#if (ELT_LESSTHAN_DATA_TYPE == FLOAT16BIT)
            bool temp = LESS_THAN(InData1, InData2);
            pOut[idx] = temp ? 1 : 0;
#else
            pOut[idx] = LESS_THAN(InData1, InData2);
#endif
          }
        }
        else
        {
          /* broadcast in dims 1 or 2 in Tile1 or TIle2 */
          for (idx = 0; idx < dim1SizeOut; idx++)
          {
            InData1 = pIn1[idx];
            InData2 = pIn2[idx];
#if (ELT_LESSTHAN_DATA_TYPE == FLOAT16BIT)
            bool temp = LESS_THAN(InData1, InData2);
            pOut[idx] = temp ? 1 : 0;
#else
            pOut[idx] = LESS_THAN(InData1, InData2);
#endif
          }
        }
      }
    }
  }

  return(XAI_ERROR_STATUS());
}

