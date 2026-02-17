/*
 * Copyright (c) 2024 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#undef MAKE_NAME_IMPL
#undef MAKE_NAME
#undef MORPH_ODT_CHECK_TILE3D
#undef MORPH_ODT_SCALAR
#undef MORPH_ODT_VECTOR
#undef MORPH_OP_SA_IP
#undef MORPH_OP_SAV_XP
#undef MORPH_OP_SAPOS_FP
#undef MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ

#if OUTPUT_DATA_TYPE == SIGNED8BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S8S8)
#define MORPH_ODT_CHECK_TILE3D  XAI_CHECK_TILE3D_S8
#define MORPH_ODT_SCALAR        int8_t
#define MORPH_ODT_VECTOR        xb_vecNx8
#define MORPH_OP_SA_IP          IVP_SANX8S_IP
#define MORPH_OP_SAV_XP         IVP_SAVNX8S_XP
#define MORPH_OP_SAPOS_FP       IVP_SAPOSNX8S_FP

#define MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc, shift)  { \
    vecOut = IVP_PACKVRNX48(vecAcc, shift);                           \
}

#elif OUTPUT_DATA_TYPE == UNSIGNED8BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S8U8)
#define MORPH_ODT_CHECK_TILE3D  XAI_CHECK_TILE3D_U8
#define MORPH_ODT_SCALAR        uint8_t
#define MORPH_ODT_VECTOR        xb_vecNx8U
#define MORPH_OP_SA_IP          IVP_SANX8U_IP
#define MORPH_OP_SAV_XP         IVP_SAVNX8U_XP
#define MORPH_OP_SAPOS_FP       IVP_SAPOSNX8U_FP

#define MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc, shift)  {                  \
    vecOut = IVP_PACKVRNX48(vecAcc, shift);                                            \
    vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) UCHAR_MAX), (xb_vecNx16) 0); \
}

#elif OUTPUT_DATA_TYPE == SIGNED16BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S8S16)
#define MORPH_ODT_CHECK_TILE3D  XAI_CHECK_TILE3D_S16
#define MORPH_ODT_SCALAR        int16_t
#define MORPH_ODT_VECTOR        xb_vecNx16
#define MORPH_OP_SA_IP          IVP_SANX16_IP
#define MORPH_OP_SAV_XP         IVP_SAVNX16_XP
#define MORPH_OP_SAPOS_FP       IVP_SAPOSNX16_FP

#define MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc, shift)  { \
    vecOut = IVP_PACKVRNX48(vecAcc, shift);                           \
}

#elif OUTPUT_DATA_TYPE == UNSIGNED16BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S8U16)
#define MORPH_ODT_CHECK_TILE3D  XAI_CHECK_TILE3D_U16
#define MORPH_ODT_SCALAR        uint16_t
#define MORPH_ODT_VECTOR        xb_vecNx16U
#define MORPH_OP_SA_IP          IVP_SANX16U_IP
#define MORPH_OP_SAV_XP         IVP_SAVNX16U_XP
#define MORPH_OP_SAPOS_FP       IVP_SAPOSNX16U_FP

#define MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc, shift)  {                                      \
    xb_vecN_2x32v hvecAccEven = IVP_PACKVRNX48_0(vecAcc, shift);                                           \
    xb_vecN_2x32v hvecAccOdd  = IVP_PACKVRNX48_1(vecAcc, shift);                                           \
    hvecAccEven = IVP_MAXN_2X32(IVP_MINN_2X32(hvecAccEven, (xb_vecN_2x32v) USHRT_MAX), (xb_vecN_2x32v) 0); \
    hvecAccOdd  = IVP_MAXN_2X32(IVP_MINN_2X32(hvecAccOdd, (xb_vecN_2x32v) USHRT_MAX), (xb_vecN_2x32v) 0);  \
    xb_vecNx16U vecAccEven = IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(hvecAccEven));                   \
    xb_vecNx16U vecAccOdd  = IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(hvecAccOdd));                    \
    vecOut = IVP_SELNX16UI(vecAccOdd, vecAccEven, IVP_SELI_16B_INTERLEAVE_1_EVEN);                         \
}
#endif

/*********************** xaiDataConversion3D_AsymQ_S8IX ************************/
/* Description : P6 implementation for conversion from either of the following */
/*               1) S8_SYM to S8_ASYM                                          */
/*               2) S8_ASYM to S8_SYM                                          */
/*               3) S8_ASYM to S8_ASYM                                         */
/*               4) S8_ASYM to U8_SYM                                          */
/*               5) S8_ASYM to S16_SYM                                         */
/*               6) S8_ASYM to U16_SYM                                         */
/* Inputs      : Input Tile, fixUp, scale, shift                               */
/* Outputs     : XI Error Code                                                 */
/* InOuts      : Output Tile                                                   */
/* Assumptions : InData is signed 8bit                                         */
/*******************************************************************************/

/********************* xaiDataConversion3D_AsymQ_S8S8  *************************/
/********************* xaiDataConversion3D_AsymQ_S8U8  *************************/
/********************* xaiDataConversion3D_AsymQ_S8S16 *************************/
/********************* xaiDataConversion3D_AsymQ_S8U16 *************************/

XAI_ERR_TYPE MAKE_NAME (xaiDataConversion3D_AsymQ)(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const int16_t fixUp,
                                                   const uint16_t scale,
                                                   const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    MORPH_ODT_CHECK_TILE3D(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR((fixUp >= SHRT_MIN) && (fixUp <= SHRT_MAX), XAI_ERR_NORM, \
                    "\nfixUp = %hi, value must be greater than or equal to -32768 and less than 32768", fixUp);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int8_t *pInput            = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  MORPH_ODT_SCALAR *pOutput = (MORPH_ODT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* Vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* Loop variables */
  int32_t x, y, z;

  /* Input and Output pointers */
  xb_vecNx8 *restrict pvecIn;
  MORPH_ODT_VECTOR *restrict pvecOut;

  /* Input and Output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

  /* Accumulators */
  xb_vecNx48 vecAcc1, vecAcc2, vecAcc3, vecAcc4;

  xb_vecNx16U vecScale = (xb_vecNx16U) (scale);

  // Assuming that the "fixUpShift" value will reside with S32 range
  int32_t fixUpShift       = (fixUp << shift);
  xb_vecNx48 vecFixUpShift = fixUpShift;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S8 bit to S16 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S8 bit */
  /*      S16 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* Input and Output vectors */
    xb_vecNx16 vecInData;
    xb_vecNx16 vecOut;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* Initialize input and output data pointer */
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (MORPH_ODT_VECTOR *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

        // Initializing the 48-bit accumulator with the 32-bit "fixUpShift" value
        vecAcc1 = vecFixUpShift;
        IVP_MULUSANX16(vecAcc1, vecScale, vecInData);
        // Packing the outcome to appropriate range
        MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc1, shift);

        /* Store output data */
        MORPH_OP_SA_IP(vecOut, vaOut, pvecOut);
      }

      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

      // Initializing the 48-bit accumulator with the 32-bit "fixUpShift" value
      vecAcc1 = vecFixUpShift;
      IVP_MULUSANX16(vecAcc1, vecScale, vecInData);
      // Packing the outcome to appropriate range
      MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut, vecAcc1, shift);

      /* Store output data */
      MORPH_OP_SAV_XP(vecOut, vaOut, pvecOut, (varlen * bytesPerPixel));
      MORPH_OP_SAPOS_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn           = &pInput[(z * inTilePitch2) + x];
        MORPH_ODT_SCALAR *pOut = &pOutput[(z * outTilePitch2) + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          // Adjusting the input and output data pointers
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (MORPH_ODT_VECTOR *) (pOut + (y * outTilePitch1));

          /* Load Input data */
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);

          // Initializing the 48-bit accumulators with the 32-bit "fixUpShift" value
          vecAcc1 = vecFixUpShift;
          vecAcc2 = vecFixUpShift;
          vecAcc3 = vecFixUpShift;
          vecAcc4 = vecFixUpShift;

          IVP_MULUSANX16(vecAcc1, vecScale, vecInData0);
          IVP_MULUSANX16(vecAcc2, vecScale, vecInData1);
          IVP_MULUSANX16(vecAcc3, vecScale, vecInData2);
          IVP_MULUSANX16(vecAcc4, vecScale, vecInData3);

          // Packing the outcome to appropriate range
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut0, vecAcc1, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut1, vecAcc2, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut2, vecAcc3, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut3, vecAcc4, shift);

          /* Store output data */
          MORPH_OP_SA_IP(vecOut0, vaOut, pvecOut);
          MORPH_OP_SA_IP(vecOut1, vaOut, pvecOut);
          MORPH_OP_SA_IP(vecOut2, vaOut, pvecOut);
          MORPH_OP_SAV_XP(vecOut3, vaOut, pvecOut, (varLen * bytesPerPixel));
          MORPH_OP_SAPOS_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn           = &pInput[(z * inTilePitch2) + x];
        MORPH_ODT_SCALAR *pOut = &pOutput[(z * outTilePitch2) + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          // Adjusting the input and output data pointers
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (MORPH_ODT_VECTOR *) (pOut + (y * outTilePitch1));

          /* Load input data */
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);

          // Initializing the 48-bit accumulators with the 32-bit "fixUpShift" value
          vecAcc1 = vecFixUpShift;
          vecAcc2 = vecFixUpShift;
          vecAcc3 = vecFixUpShift;

          IVP_MULUSANX16(vecAcc1, vecScale, vecInData0);
          IVP_MULUSANX16(vecAcc2, vecScale, vecInData1);
          IVP_MULUSANX16(vecAcc3, vecScale, vecInData2);

          // Packing the outcome to appropriate range
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut0, vecAcc1, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut1, vecAcc2, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut2, vecAcc3, shift);

          /* Store output data */
          MORPH_OP_SA_IP(vecOut0, vaOut, pvecOut);
          MORPH_OP_SA_IP(vecOut1, vaOut, pvecOut);
          MORPH_OP_SAV_XP(vecOut2, vaOut, pvecOut, (varLen * bytesPerPixel));
          MORPH_OP_SAPOS_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn           = &pInput[(z * inTilePitch2) + x];
        MORPH_ODT_SCALAR *pOut = &pOutput[(z * outTilePitch2) + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          // Adjusting the input and output data pointers
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (MORPH_ODT_VECTOR *) (pOut + (y * outTilePitch1));

          /* Load input data */
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);

          // Initializing the 48-bit accumulators with the 32-bit "fixUpShift" value
          vecAcc1 = vecFixUpShift;
          vecAcc2 = vecFixUpShift;

          IVP_MULUSANX16(vecAcc1, vecScale, vecInData0);
          IVP_MULUSANX16(vecAcc2, vecScale, vecInData1);

          // Packing the outcome to appropriate range
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut0, vecAcc1, shift);
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut1, vecAcc2, shift);

          /* Store output data */
          MORPH_OP_SA_IP(vecOut0, vaOut, pvecOut);
          MORPH_OP_SAV_XP(vecOut1, vaOut, pvecOut, (varLen * bytesPerPixel));
          MORPH_OP_SAPOS_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn           = &pInput[(z * inTilePitch2) + x];
        MORPH_ODT_SCALAR *pOut = &pOutput[(z * outTilePitch2) + x];
        int32_t varLen         = (dim1Size - x);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          // Adjusting the input and output data pointers
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (MORPH_ODT_VECTOR *) (pOut + (y * outTilePitch1));

          /* Load input data */
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);

          // Initializing the 48-bit accumulator with the 32-bit "fixUpShift" value
          vecAcc1 = vecFixUpShift;
          IVP_MULUSANX16(vecAcc1, vecScale, vecInData0);

          // Packing the outcome to appropriate range
          MORPH_PACK_ROUND_CLAMP_LIMITS_ASYMQ(vecOut0, vecAcc1, shift);

          /* Store output data */
          MORPH_OP_SAV_XP(vecOut0, vaOut, pvecOut, (varLen * bytesPerPixel));
          MORPH_OP_SAPOS_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
