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
#undef MORPH_IDT_TILECHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_VECTOR
#undef MORPH_OP_PRIME
#undef MORPH_OP_LOAD_IP
#undef MORPH_OP_MUL

#if INPUT_DATA_TYPE == SIGNED16BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S16I8)
#define MORPH_IDT_TILECHECK  XAI_CHECK_TILE3D_S16
#define MORPH_IDT_SCALAR     int16_t
#define MORPH_IDT_VECTOR     xb_vecNx16
#define MORPH_OP_PRIME       IVP_LANX16_PP
#define MORPH_OP_LOAD_IP     IVP_LANX16_IP
#define MORPH_OP_MUL         IVP_MULUSNX16

#elif INPUT_DATA_TYPE == UNSIGNED16BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, U16I8)
#define MORPH_IDT_TILECHECK  XAI_CHECK_TILE3D_U16
#define MORPH_IDT_SCALAR     uint16_t
#define MORPH_IDT_VECTOR     xb_vecNx16U
#define MORPH_OP_PRIME       IVP_LANX16U_PP
#define MORPH_OP_LOAD_IP     IVP_LANX16U_IP
#define MORPH_OP_MUL         IVP_MULUUNX16
#endif

/********************* xaiDataConversion3D_S16/U16I8 ***************************/
/* Description : P6 implementation for conversion from S16/U16 to S8 / U8     */
/*               depending on Output Tile type                                */
/* Inputs      : Input Tile, scale, shift                                     */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : InData is signed/unsigned 16bit                              */
/******************************************************************************/

XAI_ERR_TYPE MAKE_NAME (xaiDataConversion3D_)(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              const uint16_t scale,
                                              const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_TILECHECK(inTile);
    XAI_CHECK_TILE3D_I8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift value = %hhu, which should be less than 32", shift);
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

  const int16_t minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8)) ? SCHAR_MIN : 0;
  const int16_t maxLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8)) ? SCHAR_MAX : UCHAR_MAX;

  /* Get Data Pointers */
  MORPH_IDT_SCALAR *pInput = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput          = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut             = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  MORPH_IDT_VECTOR * restrict pvecIn;
  xb_vecNx8U * restrict pvecOut;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 16 bit to S8/U8 bit need to done present in    */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                  */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      16 bit to S8/U8 bit need to done exist in non-contiguous memory       */
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
    MORPH_IDT_VECTOR vecInData;
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
      pvecIn  = (MORPH_IDT_VECTOR *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8U *) (pOutput + (z * outTilePitch2));

      valign vaInData = MORPH_OP_PRIME(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        MORPH_OP_LOAD_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        vecOut = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData), shift);

        vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

        /* store output data */
        IVP_SANX8U_IP(vecOut, vaOut, pvecOut);
      }
      /* load input data */
      MORPH_OP_LOAD_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      vecOut = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData), shift);

      vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

      /* store output data */
      IVP_SAVNX8U_XP(vecOut, vaOut, pvecOut, (maxLoopCount - x));
      IVP_SAPOSNX8U_FP(vaOut, pvecOut);
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
        MORPH_IDT_SCALAR * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut           = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          MORPH_IDT_VECTOR vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (MORPH_IDT_VECTOR *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);
          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData2, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut2 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData2), shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut3 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData3), shift);
          vecOut3 = IVP_MAXNX16(IVP_MINNX16(vecOut3, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut           = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          MORPH_IDT_VECTOR vecInData0, vecInData1, vecInData2;
          xb_vecNx16 vecOut0, vecOut1, vecOut2;

          pvecIn  = (MORPH_IDT_VECTOR *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);

          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut2 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData2), shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut           = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          MORPH_IDT_VECTOR vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1;

          pvecIn  = (MORPH_IDT_VECTOR *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);

          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut           = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          MORPH_IDT_VECTOR vecInData0;
          xb_vecNx16 vecOut0;

          pvecIn  = (MORPH_IDT_VECTOR *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);

          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SAVNX8U_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
