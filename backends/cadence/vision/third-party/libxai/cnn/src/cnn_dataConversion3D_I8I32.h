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
#undef MORPH_IDT_VECTORI8
#undef MORPH_OP_PRIME
#undef MORPH_OP_LOAD_IP
#undef MORPH_OP_MUL

#if INPUT_DATA_TYPE == SIGNED8BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, S8I32)
#define MORPH_IDT_TILECHECK  XAI_CHECK_TILE3D_S8
#define MORPH_IDT_SCALAR     int8_t
#define MORPH_IDT_VECTOR     xb_vecNx16
#define MORPH_IDT_VECTORI8   xb_vecNx8
#define MORPH_OP_PRIME       IVP_LANX8S_PP
#define MORPH_OP_LOAD_IP     IVP_LANX8S_IP
#define MORPH_OP_MUL         IVP_MULUSNX16

#elif INPUT_DATA_TYPE == UNSIGNED8BIT

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER)  name ## _ ## MORPH_FNAME_SPECIFIER
#define MAKE_NAME(name)                              MAKE_NAME_IMPL(name, U8I32)
#define MORPH_IDT_TILECHECK  XAI_CHECK_TILE3D_U8
#define MORPH_IDT_SCALAR     uint8_t
#define MORPH_IDT_VECTOR     xb_vecNx16U
#define MORPH_IDT_VECTORI8   xb_vecNx8U
#define MORPH_OP_PRIME       IVP_LANX8U_PP
#define MORPH_OP_LOAD_IP     IVP_LANX8U_IP
#define MORPH_OP_MUL         IVP_MULUUNX16
#endif


/********************* xaiDataConversion3D_I8I32 ************************/
/* Description : P6 implementation for conversion from S8 to S32       */
/* Inputs      : Input Tile, scale, shift                              */
/* Outputs     : XI Error Code                                         */
/* InOuts      : Output Tile                                           */
/* Assumptions : InData is signed 8bit                                 */
/***********************************************************************/
/********************* xaiDataConversion3D_S8I32 ************************/
/********************* xaiDataConversion3D_U8I32 ************************/
XAI_ERR_TYPE MAKE_NAME (xaiDataConversion3D)(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_TILECHECK(inTile);
    XAI_CHECK_TILE3D_I32(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
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

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  MORPH_IDT_SCALAR *pInput = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pOutput         = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  const int32_t minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) ? INT_MIN : 0;

  /* loop variables */
  int32_t x, y, z;


  /* input and output pointers */
  MORPH_IDT_VECTORI8 *restrict pvecIn;
  xb_vecN_2x32v *restrict pvecOut;


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
  /*      I32 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    MORPH_IDT_VECTOR vecInData;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    xb_vecN_2x32v vecOutL, vecOutH;
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
      pvecIn  = (MORPH_IDT_VECTORI8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x32v *) (pOutput + (z * outTilePitch2));
      valign vaInData = MORPH_OP_PRIME(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        MORPH_OP_LOAD_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecIntRes      = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
        vecOutL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
        vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
        vecOutH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutH = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);
        /* store output data */
        IVP_SAN_2X32_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X32_IP(vecOutH, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      MORPH_OP_LOAD_IP(vecInData, vaInData, pvecIn);


      xb_vecNx48 vecIntRes = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData);

      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
      vecOutL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
      vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);

      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
      vecOutH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutH = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);

      /* store output data */
      IVP_SAVN_2X32_XP(vecOutL, vaOut, pvecOut, (varlen << 2));
      IVP_SAVN_2X32_XP(vecOutH, vaOut, pvecOut, ((varlen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
      IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */
    MORPH_IDT_VECTOR vecInData0, vecInData1, vecInData2, vecInData3;
    xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR * pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut          = &pOutput[z * outTilePitch2 + x];
        int32_t varLen         = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (MORPH_IDT_VECTORI8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));

          valign vaInData = MORPH_OP_PRIME(pvecIn);
          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData2, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData3, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecIntRes3 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData3);

          xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));

          vecOut0L = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut0L = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut0H = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut1H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));

          vecOut2L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut2L    = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut2H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut2H    = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes3), IVP_CVT64SNX48LL(vecIntRes3));
          vecOut3L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut3L    = IVP_MAXN_2X32(vecOut3L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes3), IVP_CVT64SNX48HL(vecIntRes3));
          vecOut3H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut3H    = IVP_MAXN_2X32(vecOut3H, (xb_vecN_2x32v) minLim);



          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2H, vaOut, pvecOut);

          IVP_SAVN_2X32_XP(vecOut3L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut3H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR *pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut         = &pOutput[z * outTilePitch2 + x];
        int32_t varLen        = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (MORPH_IDT_VECTORI8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);
          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0);

          xb_vecNx48 vecIntRes1 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1);

          xb_vecNx48 vecIntRes2 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData2);

          xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0L = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut0L = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut0H = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut1H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut2L    = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut2H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut2H    = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut2L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut2H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR *pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut         = &pOutput[z * outTilePitch2 + x];
        int32_t varLen        = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (MORPH_IDT_VECTORI8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);

          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);
          MORPH_OP_LOAD_IP(vecInData1, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0);

          xb_vecNx48 vecIntRes1 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData1);

          xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0L = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut0L = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut0H = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1L    = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut1H    = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);


          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut1L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut1H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        MORPH_IDT_SCALAR *pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut         = &pOutput[z * outTilePitch2 + x];
        int32_t varLen        = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (MORPH_IDT_VECTORI8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = MORPH_OP_PRIME(pvecIn);
          /* load input data */
          MORPH_OP_LOAD_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = MORPH_OP_MUL((xb_vecNx16U) scale, vecInData0);

          xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0L = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
          vecOut0L = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut0H = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAVN_2X32_XP(vecOut0L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut0H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
