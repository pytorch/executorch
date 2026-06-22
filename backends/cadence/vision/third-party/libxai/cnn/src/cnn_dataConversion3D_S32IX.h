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

#if ((XCHAL_VISION_TYPE >= 6))

#undef MAKE_NAME_IMPL
#undef MAKE_NAME
#undef MORPH_ODT_TILECHECK
#undef MORPH_ODT_SCALAR
#undef MORPH_ODT_VECTOR
#undef MIN_VAL
#undef MAX_VAL
#undef MORPH_STORE_SA_IP
#undef MORPH_STORE_SAV_XP
#undef MORPH_FLUSH_SAPOS
#undef BytesPerPixel

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## MORPH_FNAME_SPECIFIER

#if ((INPUT_DATA_TYPE == SIGNED32BIT) && (OUTPUT_DATA_TYPE == SIGNED8BIT))
#define MAKE_NAME(name)  MAKE_NAME_IMPL(name, S8)
#define MORPH_ODT_TILECHECK  XAI_CHECK_TILE3D_S8
#define MORPH_ODT_SCALAR     int8_t
#define MORPH_ODT_VECTOR     xb_vecNx8
#define MIN_VAL              SCHAR_MIN
#define MAX_VAL              SCHAR_MAX
#define MORPH_STORE_SA_IP    IVP_SANX8S_IP
#define MORPH_STORE_SAV_XP   IVP_SAVNX8S_XP
#define MORPH_FLUSH_SAPOS    IVP_SAPOSNX8S_FP
#define BytesPerPixel        1


#elif ((INPUT_DATA_TYPE == SIGNED32BIT) && (OUTPUT_DATA_TYPE == UNSIGNED8BIT))
#define MAKE_NAME(name)  MAKE_NAME_IMPL(name, U8)
#define MORPH_ODT_TILECHECK  XAI_CHECK_TILE3D_U8
#define MORPH_ODT_SCALAR     uint8_t
#define MORPH_ODT_VECTOR     xb_vecNx8U
#define MIN_VAL              0
#define MAX_VAL              UCHAR_MAX
#define MORPH_STORE_SA_IP    IVP_SANX8U_IP
#define MORPH_STORE_SAV_XP   IVP_SAVNX8U_XP
#define MORPH_FLUSH_SAPOS    IVP_SAPOSNX8U_FP
#define BytesPerPixel        1

#elif ((INPUT_DATA_TYPE == SIGNED32BIT) && (OUTPUT_DATA_TYPE == SIGNED16BIT))
#define MAKE_NAME(name)  MAKE_NAME_IMPL(name, S16)
#define MORPH_ODT_TILECHECK  XAI_CHECK_TILE3D_S16
#define MORPH_ODT_SCALAR     int16_t
#define MORPH_ODT_VECTOR     xb_vecNx16
#define MIN_VAL              SHRT_MIN
#define MAX_VAL              SHRT_MAX
#define MORPH_STORE_SA_IP    IVP_SANX16_IP
#define MORPH_STORE_SAV_XP   IVP_SAVNX16_XP
#define MORPH_FLUSH_SAPOS    IVP_SAPOSNX16_FP
#define BytesPerPixel        2

#elif ((INPUT_DATA_TYPE == SIGNED32BIT) && (OUTPUT_DATA_TYPE == UNSIGNED16BIT))
#define MAKE_NAME(name)  MAKE_NAME_IMPL(name, U16)
#define MORPH_ODT_TILECHECK  XAI_CHECK_TILE3D_U16
#define MORPH_ODT_SCALAR     uint16_t
#define MORPH_ODT_VECTOR     xb_vecNx16U
#define MIN_VAL              0
#define MAX_VAL              USHRT_MAX
#define MORPH_STORE_SA_IP    IVP_SANX16U_IP
#define MORPH_STORE_SAV_XP   IVP_SAVNX16U_XP
#define MORPH_FLUSH_SAPOS    IVP_SAPOSNX16U_FP
#define BytesPerPixel        2
#endif


/********************* xaiDataConversion3D_S32IX ******************************/
/* Description : P6 implementation for conversion from S32 to S8 /U8/S16/U16  */
/*               depending on Output Tile type                                */
/* Inputs      : Input Tile, scale, shift                                     */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : InData is signed 32 bit                                      */
/******************************************************************************/
/********************* xaiDataConversion3D_S32S8  *****************************/
/********************* xaiDataConversion3D_S32U8  *****************************/
/********************* xaiDataConversion3D_S32S16 ******************************/
/********************* xaiDataConversion3D_S32U16 *****************************/
XAI_ERR_TYPE MAKE_NAME (xaiDataConversion3D_S32)(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S32(inTile);
    MORPH_ODT_TILECHECK(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
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

  int32_t minLim = MIN_VAL;
  int32_t maxLim = MAX_VAL;

  /* Get Data Pointers */
  int32_t *pInput           = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  MORPH_ODT_SCALAR *pOutput = (MORPH_ODT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(outTile);

  valign vaOut = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH / 2;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecN_2x32v * restrict pvecIn;
  MORPH_ODT_VECTOR * restrict pvecOut;

  xb_vecN_2x64w vec0scaledIn64B, vec1scaledIn64B;

  /* SCALE*/
  xb_vecNx16U vecScale = (xb_vecNx16U) (scale);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 32 bit to S8/U8 bit need to done present in    */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                           */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      32 bit to S8/U8 bit need to done exist in non-contiguous memory       */
  /*      location. In order to do vectorization across first dimension, */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecN_2x32v vecInData0, vecInData1;

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
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecN_2x32v *) (pInput + (z * inTilePitch2));
      pvecOut = (MORPH_ODT_VECTOR *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LAN_2X32_PP(pvecIn);
      xb_vecNx16 vecOut, vecOut0, vecOut1;
      x = 0;
      for (; x < maxLoopCount - vectorizationWidth2X; x += vectorizationWidth2X)
      {
        /* Load input data */
        IVP_LAN_2X32_IP(vecInData0, vaInData, pvecIn);
        IVP_LAN_2X32_IP(vecInData1, vaInData, pvecIn);

        /* Multiply U16 scale with S32 input and store in 64-bit wide vector */
        vec0scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData0);
        vec1scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData1);

        /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
        xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
        xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

        /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
         * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
        vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
        vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

        /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
        vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

        /* store output data */
        MORPH_STORE_SA_IP(vecOut, vaOut, pvecOut);
      }

      /* Load remaining input data */
      IVP_LAVN_2X32_XP(vecInData0, vaInData, pvecIn, (maxLoopCount - x) * 4);
      IVP_LAVN_2X32_XP(vecInData1, vaInData, pvecIn, ((maxLoopCount - x) - (vectorizationWidth >> 1)) * 4);

      /* Multiply U16 scale with S32 input and store in 64-bit wide vector */
      vec0scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData0);
      vec1scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData1);

      /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
      xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
      xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

      /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
       * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
      vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
      vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

      /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
      vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

      /* store output data */
      MORPH_STORE_SAV_XP(vecOut, vaOut, pvecOut, (maxLoopCount - x) * BytesPerPixel);
      MORPH_FLUSH_SAPOS(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)                 /* along 3rd dimension */
    {
      x = 0;
      for (; x < dim1Size; x += vectorizationWidth2X) /* Load two vectors along 1st dimension*/
      {
        /* Initialize input and output data pointer */
        int32_t * pIn          = &pInput[z * inTilePitch2 + x];
        MORPH_ODT_SCALAR *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - x;

        for (y = 0; y < dim2Size; y++)              /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecN_2x32v vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1, vecOut;

          pvecIn  = (xb_vecN_2x32v *) (pIn + (y * inTilePitch1));
          pvecOut = (MORPH_ODT_VECTOR *) (pOut + (y * outTilePitch1));

          /* Load input data */
          valign vaInData = IVP_LAN_2X32_PP(pvecIn);
          IVP_LAVN_2X32_XP(vecInData0, vaInData, pvecIn, varLen * 4);
          IVP_LAVN_2X32_XP(vecInData1, vaInData, pvecIn, (varLen - (vectorizationWidth >> 1)) * 4);

          /* Multiply U16 scale with S32 input and store in 64-bit wide vector */
          vec0scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData0);
          vec1scaledIn64B = IVP_MULUSN_2X16X32_0(vecScale, vecInData1);

          /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
          xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
          xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

          /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
           * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
          vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
          vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

          /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
          vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

          /* Store output data */
          MORPH_STORE_SAV_XP(vecOut, vaOut, pvecOut, varLen * BytesPerPixel);
          MORPH_FLUSH_SAPOS(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //#if ((XCHAL_VISION_TYPE >= 6))


