/*
 * Copyright (c) 2022 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#include "xai_cnn.h"
#include "xai_intrin.h"

#if ((XCHAL_VISION_TYPE >= 6))

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER_IDT)               name ## _ ## MORPH_FNAME_SPECIFIER_IDT
#define MAKE_NAME_IMPL_1(name, MORPH_FNAME_SPECIFIER_IDT, dataOrder)  name ## _ ## MORPH_FNAME_SPECIFIER_IDT ## _ ## dataOrder

#if INPUT_DATA_TYPE == INTEGER8BIT

#undef MAKE_ARGUMENTS
#undef MAKE_ARGUMENTS2
#undef MAKE_NAME
#undef MAKE_NAME_1
#undef MORPH_OP_FUNCTION
#undef MORPH_OP_FUNCTION_CONST
#undef MORPH_IDT_CHECK
#undef MORPH_ADT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_FILLTILE
#undef MORPH_OP_LOAD
#undef MORPH_OP_AND
#undef MORPH_OP_SEQ
#undef MORPH_OP_SEL
#undef MORPH_OP_STORE
#undef MORPH_OP_PRIME
#undef MORPH_IDT_VEC
#undef MORPH_OP_FLUSH
#undef MORPH_VECTORIZATIONWIDTH

#define MAKE_ARGUMENTS(a, b, c)   (xai_pTile3D a, const int32_t b, xai_size3D c)
#define MAKE_ARGUMENTS2(a, b, c)  (xai_pTile3D a, const int8_t * b, xai_size3D c)
#define MORPH_OP_FUNCTION        extendWHEdges3D_I8
#define MORPH_OP_FUNCTION_CONST  extendEdgesConst3D_I8
#define MAKE_NAME(name)               MAKE_NAME_IMPL(name, I8)
#define MAKE_NAME_1(name, dataOrder)  MAKE_NAME_IMPL_1(name, I8, dataOrder)
#define MORPH_IDT_CHECK           XAI_CHECK_TILE3D_I8
#define MORPH_ADT_CHECK           XAI_CHECK_ARRAY_I8
#define MORPH_IDT_SCALAR          int8_t
#define MORPH_IDT_FILLTILE        xaiFillTile3D_I8
#define MORPH_OP_LOAD             IVP_LAV2NX8_XP
#define MORPH_OP_AND              IVP_AND2NX8
#define MORPH_OP_SEQ              IVP_SEQ2NX8
#define MORPH_OP_SEL              IVP_SEL2NX8
#define MORPH_OP_STORE            IVP_SAV2NX8_XP
#define MORPH_OP_PRIME            IVP_LA2NX8_PP
#define MORPH_IDT_VEC             xb_vec2Nx8
#define MORPH_OP_FLUSH            IVP_SAPOS2NX8_FP
#define MORPH_VECTORIZATIONWIDTH  2 * XCHAL_IVPN_SIMD_WIDTH

#elif INPUT_DATA_TYPE == INTEGER16BIT

#undef MAKE_ARGUMENTS
#undef MAKE_ARGUMENTS2
#undef MAKE_NAME
#undef MAKE_NAME_1
#undef MORPH_OP_FUNCTION
#undef MORPH_OP_FUNCTION_CONST
#undef MORPH_IDT_CHECK
#undef MORPH_ADT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_FILLTILE
#undef MORPH_OP_LOAD
#undef MORPH_OP_AND
#undef MORPH_OP_SEQ
#undef MORPH_OP_SEL
#undef MORPH_OP_STORE
#undef MORPH_OP_PRIME
#undef MORPH_IDT_VEC
#undef MORPH_OP_FLUSH
#undef MORPH_VECTORIZATIONWIDTH

#define MAKE_ARGUMENTS(a, b, c)       (xai_pTile3D a, const int32_t b, xai_size3D c)
#define MAKE_ARGUMENTS2(a, b, c)      (xai_pTile3D a, const int16_t * b, xai_size3D c)
#define MAKE_NAME(name)               MAKE_NAME_IMPL(name, I16)
#define MAKE_NAME_1(name, dataOrder)  MAKE_NAME_IMPL_1(name, I16, dataOrder)
#define MORPH_OP_FUNCTION         extendWHEdges3D_I16
#define MORPH_OP_FUNCTION_CONST   extendEdgesConst3D_I16
#define MORPH_IDT_CHECK           XAI_CHECK_TILE3D_I16
#define MORPH_ADT_CHECK           XAI_CHECK_ARRAY_I16
#define MORPH_IDT_SCALAR          int16_t
#define MORPH_IDT_FILLTILE        xaiFillTile3D_I16
#define MORPH_OP_LOAD             IVP_LAVNX16_XP
#define MORPH_OP_AND              IVP_ANDNX16
#define MORPH_OP_SEQ              IVP_SEQNX16
#define MORPH_OP_SEL              IVP_SELNX16
#define MORPH_OP_STORE            IVP_SAVNX16_XP
#define MORPH_OP_PRIME            IVP_LANX16_PP
#define MORPH_IDT_VEC             xb_vecNx16
#define MORPH_OP_FLUSH            IVP_SAPOSNX16_FP
#define MORPH_VECTORIZATIONWIDTH  XCHAL_IVPN_SIMD_WIDTH

#elif INPUT_DATA_TYPE == FLOAT16BIT
#undef MAKE_ARGUMENTS
#undef MAKE_ARGUMENTS2
#undef MAKE_NAME
#undef MAKE_NAME_1
#undef MORPH_OP_FUNCTION
#undef MORPH_OP_FUNCTION_CONST
#undef MORPH_IDT_CHECK
#undef MORPH_ADT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_FILLTILE
#undef MORPH_OP_LOAD
#undef MORPH_OP_STORE
#undef MORPH_OP_PRIME
#undef MORPH_IDT_VEC
#undef MORPH_OP_FLUSH
#undef MORPH_VECTORIZATIONWIDTH

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
#define MAKE_ARGUMENTS(a, b, c)   (xai_pTile3D a, const xb_f16 b, xai_size3D c)
#define MAKE_ARGUMENTS2(a, b, c)  (xai_pTile3D a, const xb_f16 * b, xai_size3D c)
#define MORPH_OP_FUNCTION        extendWHEdges3D_F16
#define MORPH_OP_FUNCTION_CONST  extendEdgesConst3D_F16
#define MAKE_NAME(name)               MAKE_NAME_IMPL(name, F16)
#define MAKE_NAME_1(name, dataOrder)  MAKE_NAME_IMPL_1(name, F16, dataOrder)
#define MORPH_IDT_CHECK           XAI_CHECK_TILE3D_F16
#define MORPH_ADT_CHECK           XAI_CHECK_ARRAY_F16
#define MORPH_IDT_SCALAR          xb_f16
#define MORPH_IDT_FILLTILE        xaiFillTile3D_F16
#define MORPH_OP_LOAD             IVP_LAVNXF16_XP
#define MORPH_OP_STORE            IVP_SAVNXF16_XP
#define MORPH_OP_PRIME            IVP_LANXF16_PP
#define MORPH_IDT_VEC             xb_vecNxf16
#define MORPH_OP_FLUSH            IVP_SAPOSNXF16_FP
#define MORPH_VECTORIZATIONWIDTH  XCHAL_IVPN_SIMD_WIDTH
#endif

#elif INPUT_DATA_TYPE == FLOAT32BIT
#undef MAKE_ARGUMENTS
#undef MAKE_ARGUMENTS2
#undef MAKE_NAME
#undef MAKE_NAME_1
#undef MORPH_OP_FUNCTION
#undef MORPH_OP_FUNCTION_CONST
#undef MORPH_IDT_CHECK
#undef MORPH_ADT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_FILLTILE
#undef MORPH_OP_LOAD
#undef MORPH_OP_STORE
#undef MORPH_OP_PRIME
#undef MORPH_IDT_VEC
#undef MORPH_OP_FLUSH
#undef MORPH_VECTORIZATIONWIDTH

#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#define MAKE_ARGUMENTS(a, b, c)   (xai_pTile3D a, const float b, xai_size3D c)
#define MAKE_ARGUMENTS2(a, b, c)  (xai_pTile3D a, const float * b, xai_size3D c)
#define MORPH_OP_FUNCTION        extendWHEdges3D_F32
#define MORPH_OP_FUNCTION_CONST  extendEdgesConst3D_F32
#define MAKE_NAME(name)               MAKE_NAME_IMPL(name, F32)
#define MAKE_NAME_1(name, dataOrder)  MAKE_NAME_IMPL_1(name, F32, dataOrder)
#define MORPH_IDT_CHECK           XAI_CHECK_TILE3D_F32
#define MORPH_ADT_CHECK           XAI_CHECK_ARRAY_F32
#define MORPH_IDT_SCALAR          float
#define MORPH_IDT_FILLTILE        xaiFillTile3D_F32
#define MORPH_OP_LOAD             IVP_LAVN_2XF32_XP
#define MORPH_OP_STORE            IVP_SAVN_2XF32_XP
#define MORPH_OP_PRIME            IVP_LAN_2XF32_PP
#define MORPH_IDT_VEC             xb_vecN_2xf32
#define MORPH_OP_FLUSH            IVP_SAPOSN_2XF32_FP
#define MORPH_VECTORIZATIONWIDTH  XCHAL_IVPN_SIMD_WIDTH / 2
#endif
#endif


/*====================================================================================*/
/*============= START of xaiExtendEdgesConst3D_* routines ============================*/
/*====================================================================================*/

/*************************** extendEdgesConst3D_I8  *************************/
/*************************** extendEdgesConst3D_I16 *************************/
/*************************** extendEdgesConst3D_F16 *************************/
/*************************** extendEdgesConst3D_F32 *************************/
/* Description : P6 implementation for extending the edges of a 3D tile     */
/*               with a constant value. This function extends edges across  */
/*               dimension 1 & dimension2 of  a 3D tile                     */
/* Inputs      : constant value to fill the edges                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Destination Tile                                           */
/* Assumptions : dstData is signed 8/16 bit Interger or half precision      */
/*               float(FP16) or single precision float(FP32)                */
/*               based on MORPH specifier.                                  */
/****************************************************************************/
static _XAI_INLINE_ void MAKE_NAME(extendEdgesConst3D) MAKE_ARGUMENTS(dstTile, constValue, frame3DSize)
{
  /* Getting parameters from the tile structures */
  const int32_t dim1Size  = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size  = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim1Edge1 = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2 = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1 = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2 = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  int32_t dim3Size        = XAI_TILE3D_GET_DIM3(dstTile);

  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);
  int32_t frame_dim1          = frame3DSize.dim1Size;
  int32_t frame_dim2          = frame3DSize.dim2Size;
  int32_t dim1ExtendEdgeSize  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t dim2ExtendEdgeSize  = (dim2Size + dim2Edge1 + dim2Edge2) * dstDataPitch1;

  int32_t start_x = XAI_TILE3D_GET_DIM1_COORD(dstTile);
  int32_t start_y = XAI_TILE3D_GET_DIM2_COORD(dstTile);

  MORPH_IDT_SCALAR *restrict pDst3D = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);
  int32_t ixmin                     = MAX2(start_x - dim1Edge1, 0);
  int32_t ixmax                     = MIN2(start_x + dim1Size + dim1Edge2 - 1, frame_dim1 - 1);
  int32_t iymin                     = MAX2(start_y - dim2Edge1, 0);
  int32_t iymax                     = MIN2(start_y + dim2Size + dim2Edge2 - 1, frame_dim2 - 1);

  int x, y, z; /* Loop variables */
  const MORPH_IDT_SCALAR value = constValue;

  // horizontal top
  int32_t horTopXcord  = -dim1Edge1;
  int32_t horTopYcord  = -dim2Edge1;
  int32_t horTopWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horTopHeight = iymin - (start_y - dim2Edge1);

  // horizontal bottom
  int32_t horBottomXcord  = -dim1Edge1;
  int32_t horBottomYcord  = iymax + 1 - start_y;
  int32_t horBottomWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horBottomHeight = start_y + dim2Size + dim2Edge2 - 1 - iymax;

  // vertical left
  int32_t verLeftXcord  = -dim1Edge1;
  int32_t verLeftYcord  = horTopYcord + horTopHeight;
  int32_t verLeftWidth  = ixmin - (start_x - dim1Edge1);
  int32_t verLeftHeight = iymax - iymin + 1;

  // vertical right
  int32_t verRightXcord  = ixmax + 1 - start_x;
  int32_t verRightYcord  = horTopYcord + horTopHeight;
  int32_t verRightWidth  = start_x + dim1Size + dim1Edge2 - 1 - ixmax;
  int32_t verRightHeight = iymax - iymin + 1;

  valign vaOutData1 = IVP_ZALIGN();

  MORPH_IDT_VEC *restrict pdvecOut1, *restrict pdvecOut2;
  MORPH_IDT_SCALAR *restrict pDst1, *restrict pDst2;
  /* Most optimal case is when -
     i. dim1 (including edges) has no extra padding
     ii. Each plane, i.e. dim1 * dim2 (including edges in both dimensions) has no extra padding
   */
  if ((dstDataPitch1 == dim1ExtendEdgeSize) && (dstDataPitch2 == dim2ExtendEdgeSize))
  {
    int numIter = horTopWidth * horTopHeight + horBottomWidth * horBottomHeight;

    // horizontal top first(z = 0) plane
    if (horTopHeight > 0)
    {
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      pdvecOut1 = (MORPH_IDT_VEC *) (pDst1);
      for (x = 0; x < horTopWidth * horTopHeight; x += MORPH_VECTORIZATIONWIDTH)
      {
        MORPH_OP_STORE(value, vaOutData1, pdvecOut1,
                       sizeof(MORPH_IDT_SCALAR) * (horTopWidth * horTopHeight - x));
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
      }
    } //if( horTopHeight > 0)
    z = 0;
    if (dim3Size > 1)
    {
      for (; z < dim3Size - 1; z++) // In one loop, "horizontal bottom z plane" and "horizontal top (z + 1)" plane is covered
      {
        pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
                ((horBottomYcord * dstDataPitch1) + horBottomXcord);

        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1);
        for (x = 0; x < numIter; x += MORPH_VECTORIZATIONWIDTH)
        {
          MORPH_OP_STORE(value, vaOutData1, pdvecOut1,
                         sizeof(MORPH_IDT_SCALAR) * (numIter - x));
          MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
        }
      }
    }

    // horizontal bottom last(z = dim3Size - 1) plane
    if (horBottomHeight > 0)
    {
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      pdvecOut1 = (MORPH_IDT_VEC *) (pDst1);
      for (x = 0; x < horBottomWidth * horBottomHeight; x += MORPH_VECTORIZATIONWIDTH)
      {
        MORPH_OP_STORE(value, vaOutData1, pdvecOut1,
                       sizeof(MORPH_IDT_SCALAR) * (horBottomWidth * horBottomHeight - x));
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
      }
    }
  }
  else
  {
    for (z = 0; z < dim3Size; z += 2)
    {
      int32_t remZ = XT_SALT(1, dim3Size - z);  //remaining (dim3Size - z) greater than 1, then remZ = 1, else 0

      // horizontal top
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      if (horTopHeight > 0)
      {
        for (x = 0; x < horTopWidth; x += MORPH_VECTORIZATIONWIDTH)
        {
          int32_t remX = XT_MIN((horTopWidth - x), MORPH_VECTORIZATIONWIDTH);
          for (y = 0; y < horTopHeight; y++)
          {
            pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
            pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
            MORPH_OP_STORE(value, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
            MORPH_OP_STORE(value, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
          }
        }
      } //if( horTopHeight > 0)

      // horizontal bottom
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      if (horBottomHeight > 0)
      {
        for (x = 0; x < horBottomWidth; x += MORPH_VECTORIZATIONWIDTH)
        {
          int32_t remX = XT_MIN((horBottomWidth - x), MORPH_VECTORIZATIONWIDTH);
          for (y = 0; y < horBottomHeight; y++)
          {
            pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
            pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
            MORPH_OP_STORE(value, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
            MORPH_OP_STORE(value, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
          }
        }
      }
    }
  }

  for (z = 0; z < dim3Size; z += 2)
  {
    int remZ = XT_SALT(1, dim3Size - z);  //remaining (dim3Size - z) greater than 1, then remZ = 1, else 0

    // vertical left
    pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
            ((verLeftYcord * dstDataPitch1) + verLeftXcord);
    pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
            ((verLeftYcord * dstDataPitch1) + verLeftXcord);

    for (x = 0; x < verLeftWidth; x += MORPH_VECTORIZATIONWIDTH)
    {
      int32_t remX = XT_MIN((verLeftWidth - x), MORPH_VECTORIZATIONWIDTH);
      for (y = 0; y < verLeftHeight; y++)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
        MORPH_OP_STORE(value, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
        MORPH_OP_STORE(value, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
      }
    }

    // vertical right
    pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
            ((verRightYcord * dstDataPitch1) + verRightXcord);
    pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
            ((verRightYcord * dstDataPitch1) + verRightXcord);

    for (x = 0; x < verRightWidth; x += MORPH_VECTORIZATIONWIDTH)
    {
      int32_t remX = XT_MIN((verRightWidth - x), MORPH_VECTORIZATIONWIDTH);
      for (y = 0; y < verRightHeight; y++)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
        MORPH_OP_STORE(value, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
        MORPH_OP_STORE(value, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
      }
    }
  }
}

/************************** xaiExtendEdgesConst3D_I8 ********************************/
/************************** xaiExtendEdgesConst3D_I16 *******************************/
/************************** xaiExtendEdgesConst3D_F16 *******************************/
/************************** xaiExtendEdgesConst3D_F32 *******************************/
/* Description : P6 optimized generic implementation of xaiExtendEdgesConst 3D      */
/*               function. Based on MORPH pre-processor specifiers, code            */
/*               implementation is generated during preprocessing stage. This       */
/*               method implements xaiExtendEdgesConst_I8, xaiExtendEdgesConst_I16  */
/*               xaiExtendEdgesConst3D_F16 & xaiExtendEdgesConst3D_F32 functionality*/
/* Inputs      : constant value to fill the edges                                   */
/* Outputs     : XI Error Code                                                      */
/* InOuts      : Destination Tile                                                   */
/* Assumptions : OutData is signed 8/16 bit Interger or half precision float(FP16)  */
/*               single precision float(FP32) based on MORPH specifier              */
/************************************************************************************/

XAI_ERR_TYPE MAKE_NAME(xaiExtendEdgesConst3D) MAKE_ARGUMENTS(dstTile, value, frame3DSize)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_CHECK(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
    XAI_CHECK_ERROR((frame3DSize.dim1Size > 0) && (frame3DSize.dim2Size > 0) &&                                                               \
                    (frame3DSize.dim3Size > 0), XAI_ERR_DATASIZE,                                                                             \
                    "\nframe3DSize.dim1Size = %d, frame3DSize.dim2Size = %d, frame3DSize.dim3Size = %d\nDimensions should be greater than 0", \
                    frame3DSize.dim1Size, frame3DSize.dim2Size, frame3DSize.dim3Size);
  }

  /* Getting parameters from the tile structures */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(dstTile);
  const int32_t dim1Edge1     = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2     = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1     = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2     = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  const int32_t dim3Edge1     = XAI_TILE3D_GET_DIM3_EDGE1(dstTile);
  const int32_t dim3Edge2     = XAI_TILE3D_GET_DIM3_EDGE2(dstTile);
  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);

  MORPH_IDT_SCALAR *pDst = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);

  int32_t frame_dim1 = frame3DSize.dim1Size;
  int32_t frame_dim2 = frame3DSize.dim2Size;
  int32_t frame_dim3 = frame3DSize.dim3Size;
  int32_t start_x    = XAI_TILE3D_GET_DIM1_COORD(dstTile);
  int32_t start_y    = XAI_TILE3D_GET_DIM2_COORD(dstTile);
  int32_t start_z    = XAI_TILE3D_GET_DIM3_COORD(dstTile);

  int32_t ixmin = MAX2(start_x - dim1Edge1, 0);
  int32_t ixmax = MIN2(start_x + dim1Size + dim1Edge2 - 1, frame_dim1 - 1);
  int32_t iymin = MAX2(start_y - dim2Edge1, 0);
  int32_t iymax = MIN2(start_y + dim2Size + dim2Edge2 - 1, frame_dim2 - 1);
  int32_t izmin = MAX2(start_z - dim3Edge1, 0);
  int32_t izmax = MIN2(start_z + dim3Size + dim3Edge2 - 1, frame_dim3 - 1);

  /* nothing to extend, because tile and frame intersection is empty */
  if ((ixmin > ixmax) || (iymin > iymax) || (izmin > izmax))
  {
    return(MORPH_IDT_FILLTILE(dstTile, value, 1));
  }

  /*******************************************************************************/
  /* P6 implementation of xaiExtendEdgesConst3D is split into 3 parts.            */
  /* If pitch is equal to stride, memory location to be updated across 3rd       */
  /* dimension edges is contiguous. Hence processing across edge can be          */
  /* implemented using FillTile3D functionality. Processing across 3rd dimension */
  /* is split as front end and rear end processing. Processing across 3rd        */
  /* dimension excluding the edge is implemented similar to 2D implementation of */
  /* ExtendEdges functionality.                                                  */
  /*******************************************************************************/

  MORPH_IDT_SCALAR *pDst1;

  /* Number of 2D tiles to be processed across edge1 3rd dimension */
  int32_t dim3SizeFrontEnd = izmin - (start_z - dim3Edge1);
  /* Offset calculation for Extend Edge across 3rd dimension excluding edges */
  int32_t dim3CordMiddle = izmin - start_z;
  /* Number of 2D tiles to be processed across 3rd dimension excluding edges */
  int32_t dim3SizeMiddle = izmax - izmin + 1;
  /* Offset calculation for Extend Edge across edge 2 3rd */
  int32_t dim3CordRearEnd = izmax + 1 - start_z;
  /* Number of 2D tiles processing to Extend Edge across 3rd edge2 dimension */
  int32_t dim3SizeRearEnd = start_z + dim3Size + dim3Edge2 - 1 - izmax;

  /* Update local 3D tile structure with dstTile structure parameters. Local   */
  /* 3D tile structure is used as parameter to implement fillTile functionality */
  xai_tile3D dst_t;
  /* Update parameters for local 3D tile */
  XAI_TILE3D_SET_DIM1(&dst_t, dim1Size);
  XAI_TILE3D_SET_DIM1_PITCH(&dst_t, dstDataPitch1);
  XAI_TILE3D_SET_DIM1_EDGE1(&dst_t, dim1Edge1);
  XAI_TILE3D_SET_DIM1_EDGE2(&dst_t, dim1Edge2);
  XAI_TILE3D_SET_DIM2(&dst_t, dim2Size);
  XAI_TILE3D_SET_DIM2_PITCH(&dst_t, dstDataPitch2);
  XAI_TILE3D_SET_DIM2_EDGE1(&dst_t, dim2Edge1);
  XAI_TILE3D_SET_DIM2_EDGE2(&dst_t, dim2Edge2);
  XAI_TILE3D_SET_DIM3_EDGE1(&dst_t, 0);
  XAI_TILE3D_SET_DIM3_EDGE2(&dst_t, 0);
  XAI_TILE3D_SET_DIM1_COORD(&dst_t, start_x);
  XAI_TILE3D_SET_DIM2_COORD(&dst_t, start_y);
  XAI_TILE3D_SET_DIM3_COORD(&dst_t, start_z);
  XAI_TILE3D_SET_BUFF_PTR(&dst_t, XAI_TILE3D_GET_BUFF_PTR(dstTile));
  XAI_TILE3D_SET_BUFF_SIZE(&dst_t, XAI_TILE3D_GET_BUFF_SIZE(dstTile));
  XAI_TILE3D_SET_TYPE(&dst_t, XAI_TILE3D_GET_TYPE(dstTile));

  /***********************************************************************************/
  /* Processing across the 3rd dimension edges (edge1 and edge2)                     */
  /* Processing across 3rd dimension edge 1 is referred as Front End Processing      */
  /* Processing across 3rd dimension edge 2 is referred as Rear End Processing       */
  /* Local copy of 3D tile is declared and updated with destination tile parameters. */
  /* Size parameter across third dimension is updated based on number of 2D tiles    */
  /* to be processed across front and read end. In order to effectively use the      */
  /* SIMD capabilities xaiFillTile3D implementation is utilized.                      */
  /***********************************************************************************/
  if (dim3SizeFrontEnd > 0)
  {
    /***********************************************************************************/
    /* Front end processing : Processing along the 3rd dimension edge 1.               */
    /***********************************************************************************/

    /* update destination data pointer */
    pDst1 = &pDst[((-dim3Edge1) * dstDataPitch2)];
    XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst1);
    XAI_TILE3D_SET_DIM3(&dst_t, dim3SizeFrontEnd);
    MORPH_IDT_FILLTILE(&dst_t, value, 1);
  }
  if (dim3SizeRearEnd > 0)
  {
    /***********************************************************************************/
    /* Rear end processing : Processing along the 3rd dimension edge 2.                */
    /***********************************************************************************/

    /* update destination data pointer */
    pDst1 = &pDst[dim3CordRearEnd * dstDataPitch2];
    XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst1);
    XAI_TILE3D_SET_DIM3(&dst_t, dim3SizeRearEnd);
    MORPH_IDT_FILLTILE(&dst_t, value, 1);
  }

  /* Update destination data pointer */
  pDst1 = &pDst[(dim3CordMiddle * dstDataPitch2)];
  XAI_TILE3D_SET_DIM3(&dst_t, dim3SizeMiddle);
  XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst1);

  MORPH_OP_FUNCTION_CONST(&dst_t, value, frame3DSize);
  return(XAI_ERROR_STATUS());
}


/*====================================================================================*/
/*============= END of xaiExtendEdgesConst3D_* routines ==============================*/
/*====================================================================================*/




/*====================================================================================*/
/*============= START of xaiExtendEdges3D_* routines =================================*/
/*====================================================================================*/

/************************** extendWHEdges3D_I8  *****************************/
/************************** extendWHEdges3D_I16 *****************************/
/************************** extendWHEdges3D_F16 *****************************/
/************************** extendWHEdges3D_F32 *****************************/
/* Description : P6 implementation for extending the edges of a 3D tile     */
/*               by filling different edge values for  different depths and */
/*               extends the edges along dimension 1(W) and dimension 2(H)  */
/*               3D tile                                                    */
/* Inputs      : pValue(array of edge values)                               */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Destination Tile                                           */
/* Assumptions : dstData is signed 8/16 bit Interger or half precision      */
/*               float(FP16) or single precision float(FP32)                */
/*               based on MORPH specifier.                                  */
/****************************************************************************/
static _XAI_INLINE_ void MAKE_NAME(extendWHEdges3D) MAKE_ARGUMENTS2(dstTile, pValue, frame3DSize)
{
  /* Getting parameters from the tile structures */
  const int32_t dim1Size  = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size  = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim1Edge1 = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2 = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1 = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2 = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  int32_t dim3Size        = XAI_TILE3D_GET_DIM3(dstTile);

  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);
  int32_t frame_dim1          = frame3DSize.dim1Size;
  int32_t frame_dim2          = frame3DSize.dim2Size;
  int32_t dim1ExtendEdgeSize  = dim1Size + dim1Edge1 + dim1Edge2;

  int32_t start_x = XAI_TILE3D_GET_DIM1_COORD(dstTile);
  int32_t start_y = XAI_TILE3D_GET_DIM2_COORD(dstTile);

  MORPH_IDT_SCALAR *restrict pDst3D = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);
  int32_t ixmin                     = MAX2(start_x - dim1Edge1, 0);
  int32_t ixmax                     = MIN2(start_x + dim1Size + dim1Edge2 - 1, frame_dim1 - 1);
  int32_t iymin                     = MAX2(start_y - dim2Edge1, 0);
  int32_t iymax                     = MIN2(start_y + dim2Size + dim2Edge2 - 1, frame_dim2 - 1);

  int x, y, z; /* Loop variables */

  // horizontal top
  int32_t horTopXcord  = -dim1Edge1;
  int32_t horTopYcord  = -dim2Edge1;
  int32_t horTopWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horTopHeight = iymin - (start_y - dim2Edge1);

  // horizontal bottom
  int32_t horBottomXcord  = -dim1Edge1;
  int32_t horBottomYcord  = iymax + 1 - start_y;
  int32_t horBottomWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horBottomHeight = start_y + dim2Size + dim2Edge2 - 1 - iymax;

  // vertical left
  int32_t verLeftXcord  = -dim1Edge1;
  int32_t verLeftYcord  = horTopYcord + horTopHeight;
  int32_t verLeftWidth  = ixmin - (start_x - dim1Edge1);
  int32_t verLeftHeight = iymax - iymin + 1;

  // vertical right
  int32_t verRightXcord  = ixmax + 1 - start_x;
  int32_t verRightYcord  = horTopYcord + horTopHeight;
  int32_t verRightWidth  = start_x + dim1Size + dim1Edge2 - 1 - ixmax;
  int32_t verRightHeight = iymax - iymin + 1;

  valign vaOutData1 = IVP_ZALIGN();
  valign vaOutData2 = IVP_ZALIGN();

  MORPH_IDT_VEC *restrict pdvecOut1, *restrict pdvecOut2;
  MORPH_IDT_SCALAR *restrict pDst1, *restrict pDst2;

  if (dstDataPitch1 == dim1ExtendEdgeSize)
  {
    for (z = 0; z < dim3Size; z += 2)
    {
      int32_t remZ = XT_SALT(1, dim3Size - z);  //remaining (dim3Size - z) greater than 1, then remZ = 1, else 0

      const MORPH_IDT_SCALAR value1 = pValue[z];
      const MORPH_IDT_SCALAR value2 = pValue[z + remZ];

      // horizontal top
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      if (horTopHeight > 0)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2);
        for (x = 0; x < horTopWidth * horTopHeight; x += MORPH_VECTORIZATIONWIDTH)
        {
          MORPH_OP_STORE(value1, vaOutData1, pdvecOut1,
                         sizeof(MORPH_IDT_SCALAR) * (horTopWidth * horTopHeight - x));
          MORPH_OP_FLUSH(vaOutData1, pdvecOut1);

          MORPH_OP_STORE(value2, vaOutData2, pdvecOut2,
                         sizeof(MORPH_IDT_SCALAR) * (horTopWidth * horTopHeight - x) * remZ);
          MORPH_OP_FLUSH(vaOutData2, pdvecOut2);
        }
      }

      // horizontal bottom
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      if (horBottomHeight > 0)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2);
        for (x = 0; x < horBottomWidth * horBottomHeight; x += MORPH_VECTORIZATIONWIDTH)
        {
          MORPH_OP_STORE(value1, vaOutData1, pdvecOut1,
                         sizeof(MORPH_IDT_SCALAR) * (horBottomWidth * horBottomHeight - x));
          MORPH_OP_FLUSH(vaOutData1, pdvecOut1);

          MORPH_OP_STORE(value2, vaOutData2, pdvecOut2,
                         sizeof(MORPH_IDT_SCALAR) * (horBottomWidth * horBottomHeight - x) * remZ);
          MORPH_OP_FLUSH(vaOutData2, pdvecOut2);
        }
      }
    }
  }
  else
  {
    for (z = 0; z < dim3Size; z += 2)
    {
      int32_t remZ = XT_SALT(1, dim3Size - z);  //remaining (dim3Size - z) greater than 1, then remZ = 1, else 0

      const MORPH_IDT_SCALAR value1 = pValue[z];
      const MORPH_IDT_SCALAR value2 = pValue[z + remZ];

      // horizontal top
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horTopYcord * dstDataPitch1) + horTopXcord);

      if (horTopHeight > 0)
      {
        for (x = 0; x < horTopWidth; x += MORPH_VECTORIZATIONWIDTH)
        {
          int32_t remX = XT_MIN((horTopWidth - x), MORPH_VECTORIZATIONWIDTH);
          for (y = 0; y < horTopHeight; y++)
          {
            pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
            pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
            MORPH_OP_STORE(value1, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
            MORPH_OP_STORE(value2, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
          }
        }
      } //if( horTopHeight > 0)

      // horizontal bottom
      pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);
      pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
              ((horBottomYcord * dstDataPitch1) + horBottomXcord);

      if (horBottomHeight > 0)
      {
        for (x = 0; x < horBottomWidth; x += MORPH_VECTORIZATIONWIDTH)
        {
          int32_t remX = XT_MIN((horBottomWidth - x), MORPH_VECTORIZATIONWIDTH);
          for (y = 0; y < horBottomHeight; y++)
          {
            pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
            pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
            MORPH_OP_STORE(value1, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
            MORPH_OP_STORE(value2, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
            MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
          }
        }
      }
    }
  }


  for (z = 0; z < dim3Size; z += 2)
  {
    int32_t remZ = XT_SALT(1, dim3Size - z);  //remaining (dim3Size - z) greater than 1, then remZ = 1, else 0

    const MORPH_IDT_SCALAR value1 = pValue[z];
    const MORPH_IDT_SCALAR value2 = pValue[z + remZ];

    // vertical left
    pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
            ((verLeftYcord * dstDataPitch1) + verLeftXcord);
    pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
            ((verLeftYcord * dstDataPitch1) + verLeftXcord);

    for (x = 0; x < verLeftWidth; x += MORPH_VECTORIZATIONWIDTH)
    {
      int32_t remX = XT_MIN((verLeftWidth - x), MORPH_VECTORIZATIONWIDTH);
      for (y = 0; y < verLeftHeight; y++)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
        MORPH_OP_STORE(value1, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
        MORPH_OP_STORE(value2, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
      }
    }

    // vertical right
    pDst1 = (MORPH_IDT_SCALAR *) pDst3D + (z * dstDataPitch2) + \
            ((verRightYcord * dstDataPitch1) + verRightXcord);
    pDst2 = (MORPH_IDT_SCALAR *) pDst3D + ((z + remZ) * dstDataPitch2) + \
            ((verRightYcord * dstDataPitch1) + verRightXcord);

    for (x = 0; x < verRightWidth; x += MORPH_VECTORIZATIONWIDTH)
    {
      int32_t remX = XT_MIN((verRightWidth - x), MORPH_VECTORIZATIONWIDTH);

      for (y = 0; y < verRightHeight; y++)
      {
        pdvecOut1 = (MORPH_IDT_VEC *) (pDst1 + (y * dstDataPitch1) + x);
        pdvecOut2 = (MORPH_IDT_VEC *) (pDst2 + (y * dstDataPitch1) + x);
        MORPH_OP_STORE(value1, vaOutData1, pdvecOut1, sizeof(MORPH_IDT_SCALAR) * remX);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut1);
        MORPH_OP_STORE(value2, vaOutData1, pdvecOut2, sizeof(MORPH_IDT_SCALAR) * remX * remZ);
        MORPH_OP_FLUSH(vaOutData1, pdvecOut2);
      }
    }
  }
}


/***************************** extendEdges3D_I8_WHD ******************************/
/***************************** extendEdges3D_I16_WHD *****************************/
/***************************** extendEdges3D_F16_WHD *****************************/
/***************************** extendEdges3D_F32_WHD *****************************/
/* Description : P6 optimized generic implementation of xaiExtendEdgesConst 3D    */
/*               function. Based on MORPH pre-processor specifiers, code         */
/*               implementation is generated during preprocessing stage. This    */
/*               method implements extendEdges3D_I8_WHD, extendEdges3D_I16_WHD,  */
/*               extendEdges3D_F16_WHD and extendEdges3D_F32_WHD functionality   */
/* Inputs      : constant value to fill the edges                                */
/* Outputs     : XI Error Code                                                   */
/* InOuts      : Destination Tile                                                */
/* Assumptions : OutData is signed/unsigned 8/16 bit Interger or                 */
/*               half precision float(FP16) or single precision float(FP32)      */
/*               based on MORPH specifier                                        */
/*********************************************************************************/

static _XAI_INLINE_ void MAKE_NAME_1(extendEdges3D, WHD) (xai_pTile3D dstTile,
                                                          const xai_pArray pArray,
                                                          xai_size3D frame3DSize)
{
  /* Getting parameters from the tile structures */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(dstTile);
  const int32_t dim1Edge1     = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2     = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1     = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2     = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  const int32_t dim3Edge1     = XAI_TILE3D_GET_DIM3_EDGE1(dstTile);
  const int32_t dim3Edge2     = XAI_TILE3D_GET_DIM3_EDGE2(dstTile);
  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);

  int32_t frame_dim1 = frame3DSize.dim1Size;
  int32_t frame_dim2 = frame3DSize.dim2Size;
  int32_t frame_dim3 = frame3DSize.dim3Size;
  int32_t start_x    = XAI_TILE3D_GET_DIM1_COORD(dstTile);
  int32_t start_y    = XAI_TILE3D_GET_DIM2_COORD(dstTile);
  int32_t start_z    = XAI_TILE3D_GET_DIM3_COORD(dstTile);

  int32_t ixmin = MAX2(start_x - dim1Edge1, 0);
  int32_t ixmax = MIN2(start_x + dim1Size + dim1Edge2 - 1, frame_dim1 - 1);
  int32_t iymin = MAX2(start_y - dim2Edge1, 0);
  int32_t iymax = MIN2(start_y + dim2Size + dim2Edge2 - 1, frame_dim2 - 1);
  int32_t izmin = MAX2(start_z - dim3Edge1, 0);
  int32_t izmax = MIN2(start_z + dim3Size + dim3Edge2 - 1, frame_dim3 - 1);

  /* Update local 3D tile structure with dstTile structure parameters. Local   */
  /* 3D tile structure is used as parameter to implement fillTile functionality */
  xai_tile3D dst_t;
  XAI_TILE3D_SET_DIM1(&dst_t, dim1Size);
  XAI_TILE3D_SET_DIM1_PITCH(&dst_t, dstDataPitch1);
  XAI_TILE3D_SET_DIM1_EDGE1(&dst_t, dim1Edge1);
  XAI_TILE3D_SET_DIM1_EDGE2(&dst_t, dim1Edge2);
  XAI_TILE3D_SET_DIM2(&dst_t, dim2Size);
  XAI_TILE3D_SET_DIM2_PITCH(&dst_t, dstDataPitch2);
  XAI_TILE3D_SET_DIM2_EDGE1(&dst_t, dim2Edge1);
  XAI_TILE3D_SET_DIM2_EDGE2(&dst_t, dim2Edge2);
  XAI_TILE3D_SET_DIM3_EDGE1(&dst_t, 0);
  XAI_TILE3D_SET_DIM3_EDGE2(&dst_t, 0);
  XAI_TILE3D_SET_DIM1_COORD(&dst_t, start_x);
  XAI_TILE3D_SET_DIM2_COORD(&dst_t, start_y);
  XAI_TILE3D_SET_DIM3_COORD(&dst_t, start_z);
  XAI_TILE3D_SET_BUFF_PTR(&dst_t, XAI_TILE3D_GET_BUFF_PTR(dstTile));
  XAI_TILE3D_SET_BUFF_SIZE(&dst_t, XAI_TILE3D_GET_BUFF_SIZE(dstTile));
  XAI_TILE3D_SET_TYPE(&dst_t, XAI_TILE3D_GET_TYPE(dstTile));

  MORPH_IDT_SCALAR *pDst         = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);
  const MORPH_IDT_SCALAR *pValue = (MORPH_IDT_SCALAR *) XAI_ARRAY_GET_DATA_PTR(pArray);
  int32_t z; /* Loop variable */
  MORPH_IDT_SCALAR *pDst1;
  MORPH_IDT_SCALAR value;

  /* Validation for Tile and Frame intersection */
  int32_t frameIntersectionFlag = ((ixmin > ixmax) || (iymin > iymax) || (izmin > izmax));

  /*********************************************************************************/
  /* P6 implementation of xaiExtendEdges3D is similar to xaiExtendEdgesConst3D       */
  /* implementation. In ExtendEdges functionality a unique value is used to        */
  /* xaiExtendEdges, in xaiExtendEdges3D implementation each 2D tile is filled       */
  /* with a value from xai_array, index by the co-ordinate position across third    */
  /* dimension. In xaiExtendEdges3D implementation processing across 3rd            */
  /* dimension edges, extendEdges need to perform for the entire 2D tile.          */
  /* xaiExtendEdges3D processing is split into 3 parts. ExtendEdges processing      */
  /* across 3rd dimension edges is split as front end and rear end processing.     */
  /* Processing across 3rd dimension excluding the edge is implemented similar to  */
  /* 2D implementation of extendEdges functionality.                               */
  /*********************************************************************************/

  if (frameIntersectionFlag)
  {
    /* If frameIntersectionFlag is enabled the tile exists outside frame boundary */
    /* and ExtendEdges need to be done on the entire 3D tile.                     */

    const int32_t dim3FillSize = dim3Size + dim3Edge1 + dim3Edge2;
    pDst1 = &pDst[((-dim3Edge1) * dstDataPitch2)];
    for (z = 0; z < dim3FillSize; z++) /* Loop across dim3 */
    {
      value = pValue[z];
      /* update destination data pointer */
      MORPH_IDT_SCALAR *pDst2 = pDst1 + (z * dstDataPitch2);
      XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst2);
      XAI_TILE3D_SET_DIM3(&dst_t, 1);
      MORPH_IDT_FILLTILE(&dst_t, value, 1);
    }
    return;
  }

  /* Number of 2D tiles to be processed across edge1 3rd dimension */
  int32_t dim3SizeFrontEnd = izmin - (start_z - dim3Edge1);
  /* Offset calculation for Extend Edge across 3rd dimension excluding edges */
  int32_t dim3CordMiddle = izmin - start_z;
  /* Number of 2D tiles to be processed across 3rd dimension excluding edges */
  int32_t dim3SizeMiddle = izmax - izmin + 1;
  /* Offset calculation for Extend Edge across edge 2 3rd */
  int32_t dim3CordRearEnd = izmax + 1 - start_z;
  /* Number of 2D tiles processing to Extend Edge across 3rd edge2 dimension */
  int32_t dim3SizeRearEnd = start_z + dim3Size + dim3Edge2 - 1 - izmax;

  /***********************************************************************************/
  /* Processing across the 3rd dimension edges (edge1 and edge2)                     */
  /* Processing across 3rd dimension edge 1 is referred as Front End Processing      */
  /* Processing across 3rd dimension edge 2 is referred as Rear End Processing       */
  /* Local copy of 3D tile is declared and updated with destination tile parameters. */
  /* Size parameter across third dimension is updated based on number of 2D tiles    */
  /* to be processed across front and read end. In order to effectively use the      */
  /* SIMD capabilities xaiFillTile3D implementation is utilized.                      */
  /***********************************************************************************/

  if (dim3SizeFrontEnd > 0)
  {
    /***********************************************************************************/
    /* Front end processing : Processing along the 3rd dimension edge 1.               */
    /***********************************************************************************/

    /* Update destination data pointer */
    pDst1 = &pDst[((-dim3Edge1) * dstDataPitch2)];
    XAI_TILE3D_SET_DIM3(&dst_t, 1);
    for (z = 0; z < dim3SizeFrontEnd; z++) /* Loop across dim3 */
    {
      value = pValue[z];
      /* update destination data pointer */
      MORPH_IDT_SCALAR *pDst2 = pDst1 + (z * dstDataPitch2);
      XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst2);
      MORPH_IDT_FILLTILE(&dst_t, value, 1);
    }
  }
  if (dim3SizeRearEnd > 0)
  {
    /***********************************************************************************/
    /* Rear end processing : Processing along the 3rd dimension edge 2.                */
    /***********************************************************************************/

    /* Update destination data pointer */
    pDst1 = &pDst[(dim3CordRearEnd * dstDataPitch2)];
    XAI_TILE3D_SET_DIM3(&dst_t, 1);
    for (z = 0; z < dim3SizeRearEnd; z++) /* Loop across dim3 */
    {
      /* update destination data pointer */
      MORPH_IDT_SCALAR *pDst2 = pDst1 + (z * dstDataPitch2);
      value = pValue[z + dim3CordRearEnd + dim3Edge1];
      XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst2);
      MORPH_IDT_FILLTILE(&dst_t, value, 1);
    }
  }

  /* Update destination data pointer */
  pDst1 = &pDst[(dim3CordMiddle * dstDataPitch2)];
  XAI_TILE3D_SET_DIM3(&dst_t, dim3SizeMiddle);

  XAI_TILE3D_SET_DATA_PTR(&dst_t, pDst1);
  MORPH_OP_FUNCTION(&dst_t, pValue + dim3CordMiddle + dim3Edge1, frame3DSize);
}

/*************************** extendEdges3D_I8_DWH *********************************/
/*************************** extendEdges3D_I16_DWH ********************************/
/*************************** extendEdges3D_F16_DWH ********************************/
/*************************** extendEdges3D_F32_DWH ********************************/
/* Description : P6 optimized generic implementation of xaiExtendEdgesConst 3D    */
/*               function. Based on MORPH pre-processor specifiers, code          */
/*               implementation is generated during preprocessing stage. This     */
/*               method implements extendEdges3D_I8_DWH and extendEdges3D_I16_DWH */
/*               extendEdges3D_F16_DWH and extendEdges3D_F32_DWH functionality.   */
/* Inputs      : constant value to fill the edges                                 */
/* Outputs     : XI Error Code                                                    */
/* InOuts      : Destination Tile                                                 */
/* Assumptions : OutData is signed/unsigned 8/16 bit Interger or                  */
/*               half precision float(FP16) or single precision float(FP32)       */
/*               based on MORPH specifier                                         */
/**********************************************************************************/

static _XAI_INLINE_ void MAKE_NAME_1(extendEdges3D, DWH) (xai_pTile3D dstTile,
                                                          const xai_pArray pArray,
                                                          xai_size3D frame3DSize)
{
  /* Getting parameters from the tile structures */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(dstTile);
  const int32_t dim1Edge1     = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2     = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1     = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2     = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  const int32_t dim3Edge1     = XAI_TILE3D_GET_DIM3_EDGE1(dstTile);
  const int32_t dim3Edge2     = XAI_TILE3D_GET_DIM3_EDGE2(dstTile);
  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);
  const int32_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(dstTile);

  int32_t frame_dim1 = frame3DSize.dim1Size;
  int32_t frame_dim2 = frame3DSize.dim2Size;
  int32_t frame_dim3 = frame3DSize.dim3Size;
  int32_t start_x    = XAI_TILE3D_GET_DIM1_COORD(dstTile); // along Depth
  int32_t start_y    = XAI_TILE3D_GET_DIM2_COORD(dstTile); // along Width
  int32_t start_z    = XAI_TILE3D_GET_DIM3_COORD(dstTile); // along Height

  int32_t ixmin = MAX2(start_x - dim1Edge1, 0);
  int32_t ixmax = MIN2(start_x + dim1Size + dim1Edge2 - 1, frame_dim1 - 1);
  int32_t iymin = MAX2(start_y - dim2Edge1, 0);
  int32_t iymax = MIN2(start_y + dim2Size + dim2Edge2 - 1, frame_dim2 - 1);
  int32_t izmin = MAX2(start_z - dim3Edge1, 0);
  int32_t izmax = MIN2(start_z + dim3Size + dim3Edge2 - 1, frame_dim3 - 1);

  // horizontal top
  int32_t horTopXcord  = -dim1Edge1;
  int32_t horTopYcord  = -dim2Edge1;
  int32_t horTopWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horTopHeight = iymin - (start_y - dim2Edge1);

  // horizontal bottom
  int32_t horBottomXcord  = -dim1Edge1;
  int32_t horBottomYcord  = iymax + 1 - start_y;
  int32_t horBottomWidth  = dim1Size + dim1Edge1 + dim1Edge2;
  int32_t horBottomHeight = start_y + dim2Size + dim2Edge2 - 1 - iymax;

  // vertical left
  int32_t verLeftXcord  = -dim1Edge1;
  int32_t verLeftYcord  = horTopYcord + horTopHeight;
  int32_t verLeftWidth  = ixmin - (start_x - dim1Edge1);
  int32_t verLeftHeight = iymax - iymin + 1;

  // vertical right
  int32_t verRightXcord  = ixmax + 1 - start_x;
  int32_t verRightYcord  = horTopYcord + horTopHeight;
  int32_t verRightWidth  = start_x + dim1Size + dim1Edge2 - 1 - ixmax;
  int32_t verRightHeight = iymax - iymin + 1;

  // front
  int32_t frontXcord  = -dim1Edge1;
  int32_t frontYcord  = horTopYcord + horTopHeight;
  int32_t frontZcord  = -dim3Edge1;
  int32_t frontDepth  = izmin - (start_z - dim3Edge1);
  int32_t frontWidth  = horTopWidth;
  int32_t frontHeight = iymax - iymin + 1;

  // rear
  int32_t rearXcord  = -dim1Edge1;
  int32_t rearYcord  = horTopYcord + horTopHeight;
  int32_t rearZcord  = izmax + 1 - start_z;
  int32_t rearDepth  = start_z + dim3Size + dim3Edge2 - 1 - izmax;
  int32_t rearWidth  = horTopWidth;
  int32_t rearHeight = iymax - iymin + 1;

  int x, y, z; /* Loop variables */
  valign vaOutData = IVP_ZALIGN();
  valign vaArray;
  int32_t vectorizationWidth = MORPH_VECTORIZATIONWIDTH;

  MORPH_IDT_SCALAR *restrict pDst3D = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);
  MORPH_IDT_SCALAR *restrict pArr   = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(pArray) + dim1Edge1;

  MORPH_IDT_VEC *restrict pdvecArr, *restrict pdvecDst;
  MORPH_IDT_VEC dvecArrData;

  /* Tile and frame intersection is empty,fill entire tile with edge values */
  if ((ixmin > ixmax) || (iymin > iymax) || (izmin > izmax))
  {
    pdvecArr = (MORPH_IDT_VEC *) (pArr - dim1Edge1);

    /* priming of pArray */
    vaArray = MORPH_OP_PRIME(pdvecArr);

    for (x = 0; x < (dim1Size + dim1Edge1 + dim1Edge2); x += vectorizationWidth)
    {
      /* Load pArray */
      MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (dim1Size + dim1Edge1 + dim1Edge2 - x) * bytesPerPixel);

      for (z = 0; z < (dim3Size + dim3Edge1 + dim3Edge2); z++)
      {
        for (y = 0; y < (dim2Size + dim2Edge1 + dim2Edge2); y++)
        {
          pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (z - dim3Edge1) * dstDataPitch2 + \
                                        (y - dim2Edge1) * dstDataPitch1 + (-dim1Edge1) + x);

          /* store array value in destination */
          MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (dim1Size + dim1Edge1 + dim1Edge2 - x) * bytesPerPixel);

          MORPH_OP_FLUSH(vaOutData, pdvecDst);
        }
      }
    }
  }
  else
  {
    /* Front Height Edge */
    if (frontDepth > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + frontXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < frontWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (frontWidth - x) * bytesPerPixel);

        for (z = 0; z < frontDepth; z++)
        {
          for (y = 0; y < frontHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (frontZcord + z) * dstDataPitch2 + \
                                          (y + frontYcord) * dstDataPitch1 + frontXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (frontWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }

    /* Rear Height Edge */
    if (rearDepth > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + rearXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < rearWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (rearWidth - x) * bytesPerPixel);

        for (z = 0; z < rearDepth; z++)
        {
          for (y = 0; y < rearHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (rearZcord + z) * dstDataPitch2 + \
                                          (y + rearYcord) * dstDataPitch1 + rearXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (rearWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }

    /* Top Width Edge */
    if (horTopHeight > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + horTopXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < horTopWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (horTopWidth - x) * bytesPerPixel);

        for (z = 0; z < (dim3Size + dim3Edge1 + dim3Edge2); z++)
        {
          for (y = 0; y < horTopHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (z - dim3Edge1) * dstDataPitch2 + \
                                          (horTopYcord + y) * dstDataPitch1 + horTopXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (horTopWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }

    /* Bottom Width Edge */
    if (horBottomHeight > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + horBottomXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < horBottomWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (horBottomWidth - x) * bytesPerPixel);

        for (z = 0; z < (dim3Size + dim3Edge1 + dim3Edge2); z++)
        {
          for (y = 0; y < horBottomHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (z - dim3Edge1) * dstDataPitch2 + \
                                          (horBottomYcord + y) * dstDataPitch1 + horBottomXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (horBottomWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }

    /* Left Depth Edge */
    if (verLeftWidth > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + verLeftXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < verLeftWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (verLeftWidth - x) * bytesPerPixel);

        for (z = 0; z < (dim3Size + dim3Edge1 + dim3Edge2); z++)
        {
          for (y = 0; y < verLeftHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (z - dim3Edge1) * dstDataPitch2 + \
                                          (verLeftYcord + y) * dstDataPitch1 + verLeftXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (verLeftWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }

    /* Right Depth Edge */
    if (verRightWidth > 0)
    {
      pdvecArr = (MORPH_IDT_VEC *) (pArr + verRightXcord);

      /* priming of pArray */
      vaArray = MORPH_OP_PRIME(pdvecArr);

      for (x = 0; x < verRightWidth; x += vectorizationWidth)
      {
        /* Load pArray */
        MORPH_OP_LOAD(dvecArrData, vaArray, pdvecArr, (verRightWidth - x) * bytesPerPixel);

        for (z = 0; z < (dim3Size + dim3Edge1 + dim3Edge2); z++)
        {
          for (y = 0; y < verRightHeight; y++)
          {
            pdvecDst = (MORPH_IDT_VEC *) (pDst3D + (z - dim3Edge1) * dstDataPitch2 + \
                                          (verRightYcord + y) * dstDataPitch1 + verRightXcord + x);

            /* store array value in destination */
            MORPH_OP_STORE(dvecArrData, vaOutData, pdvecDst, (verRightWidth - x) * bytesPerPixel);

            MORPH_OP_FLUSH(vaOutData, pdvecDst);
          }
        }
      }
    }
  }
}

#if INPUT_DATA_TYPE == INTEGER8BIT
/***********************   xaiExtendEdges3D_I8   *****************************/
/* Description : General API for ExtendEdges3D optimized implementation     */
/*               Calls one of the ExtendEdges3D functions based             */
/*               on the parameters                                          */
/* Inputs      : pArray, frame3DSize                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Input Tile                                                 */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdges3D_I8(xai_pTile3D dstTile,
                                 const xai_pArray pArray,
                                 xai_size3D frame3DSize)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_I8(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
    XAI_CHECK_ERROR(
      ((XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD) || (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)), \
      XAI_ERR_BADARG, "Provided Data Order not supported.");
    XAI_CHECK_POINTER(pArray);
    XAI_CHECK_ERROR(XAI_ARRAY_IS_CONSISTENT(pArray), XAI_ERR_BADARG, "The argument pArray is invalid");
    XAI_CHECK_ERROR((frame3DSize.dim1Size > 0) && (frame3DSize.dim2Size > 0) &&                                                             \
                    (frame3DSize.dim3Size > 0), XAI_ERR_DATASIZE,                                                                           \
                    "\nframe3DSize.dim1Size = %d, frame3DSize.dim2Size = %d, frame3DSize.dim3Size = %d\nDimensions must be greater than 0", \
                    frame3DSize.dim1Size, frame3DSize.dim2Size, frame3DSize.dim3Size);
  }
  if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM3(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM3_EDGE1(dstTile) + XAI_TILE3D_GET_DIM3_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_I8_WHD(dstTile, pArray, frame3DSize);
  }
  else if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM1(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM1_EDGE1(dstTile) + XAI_TILE3D_GET_DIM1_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_I8_DWH(dstTile, pArray, frame3DSize);
  }
  else
  {
    return(XAI_ERR_NO_VARIANT);
  }

  return(XAI_ERROR_STATUS());
}

#elif INPUT_DATA_TYPE == INTEGER16BIT
/***********************   xaiExtendEdges3D_I16   ****************************/
/* Description : General API for ExtendEdges3D optimized implementation     */
/*               Calls one of the ExtendEdges3D functions based             */
/*               on the parameters                                          */
/* Inputs      : pArray, frame3DSize                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Input Tile                                                 */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdges3D_I16(xai_pTile3D dstTile,
                                  const xai_pArray pArray,
                                  xai_size3D frame3DSize)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_X16(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
    XAI_CHECK_ERROR(
      ((XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD) || (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)), \
      XAI_ERR_BADARG, "Provided Data Order not supported.");
    XAI_CHECK_POINTER(pArray);
    XAI_CHECK_ERROR(XAI_ARRAY_IS_CONSISTENT(pArray), XAI_ERR_BADARG, "The argument pArray is invalid");
    XAI_CHECK_ERROR((frame3DSize.dim1Size > 0) && (frame3DSize.dim2Size > 0) &&                                                             \
                    (frame3DSize.dim3Size > 0), XAI_ERR_DATASIZE,                                                                           \
                    "\nframe3DSize.dim1Size = %d, frame3DSize.dim2Size = %d, frame3DSize.dim3Size = %d\nDimensions must be greater than 0", \
                    frame3DSize.dim1Size, frame3DSize.dim2Size, frame3DSize.dim3Size);
  }
  if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM3(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM3_EDGE1(dstTile) + XAI_TILE3D_GET_DIM3_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_I16_WHD(dstTile, pArray, frame3DSize);
  }
  else if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM1(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM1_EDGE1(dstTile) + XAI_TILE3D_GET_DIM1_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_I16_DWH(dstTile, pArray, frame3DSize);
  }
  else
  {
    return(XAI_ERR_NO_VARIANT);
  }

  return(XAI_ERROR_STATUS());
}

#elif INPUT_DATA_TYPE == FLOAT16BIT
#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
/***********************   xaiExtendEdges3D_F16   ****************************/
/* Description : General API for ExtendEdges3D optimized implementation     */
/*               Calls one of the ExtendEdges3D functions based             */
/*               on the parameters                                          */
/* Inputs      : pArray, frame3DSize                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Input Tile                                                 */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdges3D_F16(xai_pTile3D dstTile,
                                  const xai_pArray pArray,
                                  xai_size3D frame3DSize)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_F16(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
    XAI_CHECK_ERROR(
      ((XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD) || (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)), \
      XAI_ERR_BADARG, "Provided Data Order not supported.");
    XAI_CHECK_POINTER(pArray);
    XAI_CHECK_ERROR(XAI_ARRAY_IS_CONSISTENT(pArray), XAI_ERR_BADARG, "The argument pArray is invalid");
    XAI_CHECK_ERROR((frame3DSize.dim1Size > 0) && (frame3DSize.dim2Size > 0) &&                                                             \
                    (frame3DSize.dim3Size > 0), XAI_ERR_DATASIZE,                                                                           \
                    "\nframe3DSize.dim1Size = %d, frame3DSize.dim2Size = %d, frame3DSize.dim3Size = %d\nDimensions must be greater than 0", \
                    frame3DSize.dim1Size, frame3DSize.dim2Size, frame3DSize.dim3Size);
  }
  if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM3(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM3_EDGE1(dstTile) + XAI_TILE3D_GET_DIM3_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_F16_WHD(dstTile, pArray, frame3DSize);
  }
  else if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM1(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM1_EDGE1(dstTile) + XAI_TILE3D_GET_DIM1_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_F16_DWH(dstTile, pArray, frame3DSize);
  }
  return(XAI_ERROR_STATUS());
}
#endif //#if (XCHAL_HAVE_VISION_HP_VFPU == 1)

#elif INPUT_DATA_TYPE == FLOAT32BIT
#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
/***********************   xaiExtendEdges3D_F32   ****************************/
/* Description : General API for ExtendEdges3D optimized implementation     */
/*               Calls one of the ExtendEdges3D functions based             */
/*               on the parameters                                          */
/* Inputs      : pArray, frame3DSize                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Input Tile                                                 */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdges3D_F32(xai_pTile3D dstTile,
                                  const xai_pArray pArray,
                                  xai_size3D frame3DSize)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_F32(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
    XAI_CHECK_ERROR(
      ((XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD) || (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)), \
      XAI_ERR_BADARG, "Provided Data Order not supported.");
    XAI_CHECK_POINTER(pArray);
    XAI_CHECK_ERROR(XAI_ARRAY_IS_CONSISTENT(pArray), XAI_ERR_BADARG, "The argument pArray is invalid");
    XAI_CHECK_ERROR((frame3DSize.dim1Size > 0) && (frame3DSize.dim2Size > 0) &&                                                             \
                    (frame3DSize.dim3Size > 0), XAI_ERR_DATASIZE,                                                                           \
                    "\nframe3DSize.dim1Size = %d, frame3DSize.dim2Size = %d, frame3DSize.dim3Size = %d\nDimensions must be greater than 0", \
                    frame3DSize.dim1Size, frame3DSize.dim2Size, frame3DSize.dim3Size);
  }
  if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_WHD)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM3(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM3_EDGE1(dstTile) + XAI_TILE3D_GET_DIM3_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_F32_WHD(dstTile, pArray, frame3DSize);
  }
  else if (XAI_TILE3D_GET_DATA_ORDER(dstTile) == XAI_DWH)
  {
    XAI_ERROR_CHECKS_CONTINUE()
    {
      XAI_CHECK_ERROR(
        ((XAI_ARRAY_GET_WIDTH(pArray) >= (XAI_TILE3D_GET_DIM1(dstTile)                                                                  \
                                          + XAI_TILE3D_GET_DIM1_EDGE1(dstTile) + XAI_TILE3D_GET_DIM1_EDGE2(dstTile)))), XAI_ERR_BADARG, \
        "pArray width parameter is not set as required");
    }
    extendEdges3D_F32_DWH(dstTile, pArray, frame3DSize);
  }
  return(XAI_ERROR_STATUS());
}
#endif //#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#endif //INPUT_DATA_TYPE

/*====================================================================================*/
/*=============== END of xaiExtendEdges3D_* routines =================================*/
/*====================================================================================*/
#endif //if ((XCHAL_VISION_TYPE >= 6))
