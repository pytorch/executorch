/*
 * Copyright (c) 2021 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#if ((XCHAL_VISION_TYPE >= 6))



#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER_IDT)  name ## _ ## MORPH_FNAME_SPECIFIER_IDT

#if INPUT_DATA_TYPE == INTEGER8BIT

#undef MAKE_ARGUMENTS
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_VECTOR
#undef MORPH_VECTORIZATION_WIDTH
#undef MORPH_OP_STORE_IP
#undef MORPH_OP_VAR_STORE_XP
#undef MORPH_OP_PRIME
#undef MORPH_OP_FLUSH
#undef MORPH_BYTES_PER_PIXEL

#define MAKE_ARGUMENTS(a, b, c)  (xai_pTile3D a, const int32_t b, xai_bool c)
#define MAKE_NAME(name)          MAKE_NAME_IMPL(name, I8)
#define MORPH_IDT_CHECK            XAI_CHECK_TILE3D_I8
#define MORPH_IDT_SCALAR           int8_t
#define MORPH_IDT_VECTOR           xb_vec2Nx8
#define MORPH_VECTORIZATION_WIDTH  (2 * XCHAL_IVPN_SIMD_WIDTH)
#define MORPH_OP_STORE_IP          IVP_SA2NX8_IP
#define MORPH_OP_VAR_STORE_XP      IVP_SAV2NX8_XP
#define MORPH_OP_PRIME             IVP_LA2NX8_PP
#define MORPH_OP_FLUSH             IVP_SAPOS2NX8_FP
#define MORPH_BYTES_PER_PIXEL      1

#elif INPUT_DATA_TYPE == INTEGER16BIT

#undef MAKE_ARGUMENTS
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_VECTOR
#undef MORPH_VECTORIZATION_WIDTH
#undef MORPH_OP_STORE_IP
#undef MORPH_OP_VAR_STORE_XP
#undef MORPH_OP_PRIME
#undef MORPH_OP_FLUSH
#undef MORPH_BYTES_PER_PIXEL

#define MAKE_ARGUMENTS(a, b, c)  (xai_pTile3D a, const int32_t b, xai_bool c)
#define MAKE_NAME(name)          MAKE_NAME_IMPL(name, I16)
#define MORPH_IDT_CHECK            XAI_CHECK_TILE3D_I16
#define MORPH_IDT_SCALAR           int16_t
#define MORPH_IDT_VECTOR           xb_vecNx16
#define MORPH_VECTORIZATION_WIDTH  (XCHAL_IVPN_SIMD_WIDTH)
#define MORPH_OP_STORE_IP          IVP_SANX16_IP
#define MORPH_OP_VAR_STORE_XP      IVP_SAVNX16_XP
#define MORPH_OP_PRIME             IVP_LANX16_PP
#define MORPH_OP_FLUSH             IVP_SAPOSNX16_FP
#define MORPH_BYTES_PER_PIXEL      2

#elif INPUT_DATA_TYPE == FLOAT16BIT

#undef MAKE_ARGUMENTS
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_VECTOR
#undef MORPH_VECTORIZATION_WIDTH
#undef MORPH_OP_STORE_IP
#undef MORPH_OP_VAR_STORE_XP
#undef MORPH_OP_PRIME
#undef MORPH_OP_FLUSH
#undef MORPH_BYTES_PER_PIXEL

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
#define MAKE_ARGUMENTS(a, b, c)  (xai_pTile3D a, const xb_f16 b, xai_bool c)
#define MAKE_NAME(name)          MAKE_NAME_IMPL(name, F16)
#define MORPH_IDT_CHECK            XAI_CHECK_TILE3D_F16
#define MORPH_IDT_SCALAR           xb_f16
#define MORPH_IDT_VECTOR           xb_vecNxf16
#define MORPH_VECTORIZATION_WIDTH  (XCHAL_IVPN_SIMD_WIDTH)
#define MORPH_OP_STORE_IP          IVP_SANXF16_IP
#define MORPH_OP_VAR_STORE_XP      IVP_SAVNXF16_XP
#define MORPH_OP_PRIME             IVP_LANXF16_PP
#define MORPH_OP_FLUSH             IVP_SAPOSNXF16_FP
#define MORPH_BYTES_PER_PIXEL      2
#endif

#elif INPUT_DATA_TYPE == FLOAT32BIT

#undef MAKE_ARGUMENTS
#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_VECTOR
#undef MORPH_VECTORIZATION_WIDTH
#undef MORPH_OP_STORE_IP
#undef MORPH_OP_VAR_STORE_XP
#undef MORPH_OP_PRIME
#undef MORPH_OP_FLUSH
#undef MORPH_BYTES_PER_PIXEL

#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#define MAKE_ARGUMENTS(a, b, c)  (xai_pTile3D a, const float b, xai_bool c)
#define MAKE_NAME(name)          MAKE_NAME_IMPL(name, F32)
#define MORPH_IDT_CHECK            XAI_CHECK_TILE3D_F32
#define MORPH_IDT_SCALAR           float
#define MORPH_IDT_VECTOR           xb_vecN_2xf32
#define MORPH_VECTORIZATION_WIDTH  (XCHAL_IVPN_SIMD_WIDTH / 2)
#define MORPH_OP_STORE_IP          IVP_SAN_2XF32_IP
#define MORPH_OP_VAR_STORE_XP      IVP_SAVN_2XF32_XP
#define MORPH_OP_PRIME             IVP_LAN_2XF32_PP
#define MORPH_OP_FLUSH             IVP_SAPOSN_2XF32_FP
#define MORPH_BYTES_PER_PIXEL      4
#endif
#endif

/**************************************************************************************/
/*                                 MAKE_NAME(xaiFillTile3D)                            */
/**************************************************************************************/

/*******************************   xaiFillTile3D  *************************************/
/* Description : P6 optimized generic implementation of FillTile 3D function.         */
/*               Based on MORPH pre-processor specifiers, code implementation         */
/*               is generated during pre-processing stage. This method implements     */
/*               xaiFillTile3D_I8, xaiFillTile3D_I16, xaiFillTile3D_F16 and           */
/*               xaiFillTile3D_F32 functionality.                                     */
/* Inputs      : Constant value to fill, fill_edge_extension                          */
/* Outputs     : XI Error Code                                                        */
/* InOuts      : Output Tile                                                          */
/* Assumptions : OutData is signed 8/16 bit Integer or half precision float(FP16) or  */
/*               single precision float(FP32) based on MORPH specifier                */
/**************************************************************************************/

/****************************** xaiFillTile3D_I8 ***************************************/
/****************************** xaiFillTile3D_I16 **************************************/
/****************************** xaiFillTile3D_F16 **************************************/
/****************************** xaiFillTile3D_F32 **************************************/

XAI_ERR_TYPE MAKE_NAME(xaiFillTile3D) MAKE_ARGUMENTS(dstTile, value, fill_edge_extension)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_CHECK(dstTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(dstTile);
  }

  /* Getting parameters from the tile structures */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(dstTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(dstTile);
  const int32_t dim1Edge1     = XAI_TILE3D_GET_DIM1_EDGE1(dstTile);
  const int32_t dim1Edge2     = XAI_TILE3D_GET_DIM1_EDGE2(dstTile);
  const int32_t dim2Edge1     = XAI_TILE3D_GET_DIM2_EDGE1(dstTile);
  const int32_t dim2Edge2     = XAI_TILE3D_GET_DIM2_EDGE2(dstTile);
  const int32_t dim3Edge1     = XAI_TILE3D_GET_DIM3_EDGE1(dstTile);
  const int32_t dim3Edge2     = XAI_TILE3D_GET_DIM3_EDGE2(dstTile);
  const int32_t dstDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(dstTile);
  const int32_t dstDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(dstTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(dstTile);
  MORPH_IDT_SCALAR *pDst      = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(dstTile);

  int32_t z, x, y;
  /* Vectorization for xaiFillTile3D function is always done across the first dimension */
  int32_t vectorizationWidth = MORPH_VECTORIZATION_WIDTH;
  int32_t dim1FillSize       = dim1Size;
  int32_t dim2FillSize       = dim2Size;
  int32_t dim3FillSize       = dim3Size;
  int32_t maxLoopCount;

  MORPH_IDT_VECTOR* restrict pdvecOut;
  valign vaOutData          = IVP_ZALIGN();
  MORPH_IDT_VECTOR vecValue = value;

  /* If fill_edge_extension flag is enabled update destination data pointer  */
  /* and data fill size across all 3 dimensions.                             */

  if (fill_edge_extension)
  {
    dim1FillSize = dim1Size + dim1Edge1 + dim1Edge2;
    dim2FillSize = dim2Size + dim2Edge1 + dim2Edge2;
    dim3FillSize = dim3Size + dim3Edge1 + dim3Edge2;
    pDst         = &pDst[-dim1Edge1 + ((-dim2Edge1) * dstDataPitch1) + ((-dim3Edge1) * dstDataPitch2)];
  }

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When destination tile pitch is equal to destination tile fill size.     */
  /*    - If above condition holds good, memory location to be filled           */
  /*      with constant value is contiguous. Hence vectorization can be         */
  /*      utilized effectively                                                  */
  /* 2. When destination tile pitch is greater than destination tile fill size. */
  /*    - If above condition holds good, memory location to be filled           */
  /*      with constant value is not contiguous. In order to do                 */
  /*      vectorization across first dimension, destination data pointers       */
  /*      need to be updated based on destination tile fill size and            */
  /*      destination tile pitch                                                */
  /******************************************************************************/
  if (dstDataPitch1 == dim1FillSize)
  {
    /* Data to be filled exist in contiguous memory location with respect to */
    /* first dimension                                                       */

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3FillSize;
    maxLoopCount = dim1FillSize * dim2FillSize;
    if (dstDataPitch2 == maxLoopCount)
    {
      /* Data to be filled exist in contiguous memory location with respect to */
      /* first and second dimension                                            */

      /* Update max loop counter */
      maxLoopCount    *= dim3FillSize;
      dim3MaxLoopCount = 1;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize destination data pointer */
      pdvecOut = (MORPH_IDT_VECTOR *) (pDst + (z * dstDataPitch2));
      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
      }

      MORPH_OP_VAR_STORE_XP(vecValue, vaOutData, pdvecOut,
                            (maxLoopCount - x) * MORPH_BYTES_PER_PIXEL);
      MORPH_OP_FLUSH(vaOutData, pdvecOut);
    }
  }
  else
  {
    /* else block execute if destination tile pitch is */
    /* greater than destination tile fill size         */
    for (z = 0; z < dim3FillSize; z++) /* Loop across dim3 */
    {
      x = 0;
      /* Loop across dimension 1 */
      /* Condition check added to maximize vectorization across dimension 1*/
      /* Loop across dim1 */
      for (; x < (dim1FillSize - 3 * vectorizationWidth); x += 4 * vectorizationWidth)
      {
        /* initialize destination data pointer */
        MORPH_IDT_SCALAR *pDst1 = pDst + x + (z * dstDataPitch2);
        for (y = 0; y < dim2FillSize; y++) /* Loop across dim2 */
        {
          pdvecOut = (MORPH_IDT_VECTOR *) (pDst1 + (y * dstDataPitch1));
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_VAR_STORE_XP(vecValue, vaOutData, pdvecOut,
                                (dim1FillSize - (x + 3 * vectorizationWidth)) * MORPH_BYTES_PER_PIXEL);
          MORPH_OP_FLUSH(vaOutData, pdvecOut);
        }
      }
      if (x < (dim1FillSize - 2 * vectorizationWidth))
      {
        /* initialize destination data pointer */
        MORPH_IDT_SCALAR *pDst1 = pDst + x + (z * dstDataPitch2);
        for (y = 0; y < dim2FillSize; y++) /* Loop across dim2 */
        {
          pdvecOut = (MORPH_IDT_VECTOR *) (pDst1 + (y * dstDataPitch1));
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_VAR_STORE_XP(vecValue, vaOutData, pdvecOut,
                                (dim1FillSize - (x + 2 * vectorizationWidth)) * MORPH_BYTES_PER_PIXEL);
          MORPH_OP_FLUSH(vaOutData, pdvecOut);
        }
      }
      else if (x < (dim1FillSize - vectorizationWidth))
      {
        /* initialize destination data pointer */
        MORPH_IDT_SCALAR *pDst1 = pDst + x + (z * dstDataPitch2);
        for (y = 0; y < dim2FillSize; y++) /* Loop across dim2 */
        {
          pdvecOut = (MORPH_IDT_VECTOR *) (pDst1 + (y * dstDataPitch1));
          MORPH_OP_STORE_IP(vecValue, vaOutData, pdvecOut);
          MORPH_OP_VAR_STORE_XP(vecValue, vaOutData, pdvecOut,
                                (dim1FillSize - (x + vectorizationWidth)) * MORPH_BYTES_PER_PIXEL);
          MORPH_OP_FLUSH(vaOutData, pdvecOut);
        }
      }
      else if (x < dim1FillSize)
      {
        /* initialize destination data pointer */
        MORPH_IDT_SCALAR *pDst1 = pDst + x + (z * dstDataPitch2);
        for (y = 0; y < dim2FillSize; y++) /* Loop across dim2 */
        {
          pdvecOut = (MORPH_IDT_VECTOR *) (pDst1 + (y * dstDataPitch1));
          MORPH_OP_VAR_STORE_XP(vecValue, vaOutData, pdvecOut,
                                (dim1FillSize - x) * MORPH_BYTES_PER_PIXEL);
          MORPH_OP_FLUSH(vaOutData, pdvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
