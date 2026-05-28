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

#define VQ_TRUE   1
#define VQ_FALSE  0

#undef MAKE_NAME_VQ
#undef MAKE_ARGUMENTS
#undef MAKE_PARAMS

#if DILATED_VQ_CONV_S16 == VQ_TRUE

#define MAKE_NAME_VQ(a, b)             a ## VQ ## b
#define MAKE_ARGUMENTS(a, b, c, d, e)  (const xai_pTile3D a, const xai_pTile4D b, const xai_pArray c, const xai_pArray outputScaleArray, xai_pTile3D d, const xai_cnn_conv_params * e)
#define MAKE_PARAMS(a, b, c, d, e)     (a, b, c, outputScaleArray, d, e)

#elif DILATED_VQ_CONV_S16 == VQ_FALSE

#define MAKE_NAME_VQ(a, b)             a ## b
#define MAKE_ARGUMENTS(a, b, c, d, e)  (const xai_pTile3D a, const xai_pTile4D b, const xai_pArray c, xai_pTile3D d, const xai_cnn_conv_params * e)
#define MAKE_PARAMS(a, b, c, d, e)     (a, b, c, d, e)
#endif

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER_IDT, suffix)  name ## _ ## MORPH_FNAME_SPECIFIER_IDT ## suffix

#define MAKE_NAME(name, suffix)                                  MAKE_NAME_IMPL(name, S16, suffix)

/*********************************************************************************
 **************  xaiConvolved(VQ)3D_S_MxNj1d1_S16S16I16_MOW_WHD  *******************
 **********************************************************************************/
/*********************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution.     */
/*               Code implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN 3D dilated convolution  */
/*               function and MxN 3D VQ dilated convolution function for S16 bit */
/*               input data with input stride equal to 1                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                   */
/*               Output scale array, CNN convolution params structure            */
/* Outputs     : XI Error Code                                                   */
/* InOuts      : Output Tile                                                     */
/* Assumptions : CoeffData is S16                                                */
/*               biasArray is signed 64b, value not exceeding signed 48b         */
/*               Output scale array is U16                                       */
/*               OutData is S16 / U16                                            */
/*               Kernel Size is MxNxDxN                                          */
/*               Input and Output are in WHD format                              */
/*               Coeff is in WHDN format                                         */
/*********************************************************************************/

/****************** xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD *********************/
/****************** xaiConvolvedVQ3D_S_MxNj1d1_S16S16I16_MOW_WHD *******************/

XAI_ERR_TYPE MAKE_NAME(MAKE_NAME_VQ(xaiConvolved, 3D_S_MxNj1d1), S16I16_MOW_WHD) MAKE_ARGUMENTS(inTile, coeffTile, biasArray, outTile, param)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_I16(outTile);
    XAI_CHECK_TILE4D_S16(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S64(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTile) <= 16) &&                   \
                    (XAI_TILE4D_GET_DIM2(coeffTile) <= 16),                     \
                    XAI_ERR_KSIZE, "\nKernel Width = %d and Kernel Height = %d\n \
                    Kernel Width or Height should be less than or equal to 16", \
                    XAI_TILE4D_GET_DIM1(coeffTile), XAI_TILE4D_GET_DIM2(coeffTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)), \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and Stride along height = %hhu\n \
                     Stride along width should be equal to stride along height",          \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_STRIDE(param, 1);
    XAI_CHECK_DILATION(param, 1);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and Dilation along height = %hhu\n \
                     Dilation along width should be equal to dilation along height",
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_EDGES_MOW_WHD(inTile, coeffTile, param);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_WHDN);
    XAI_CHECK_CONSISTENCY_MOW_WHD(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_COEFFTILE_CONTIGUOUS(coeffTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 32, \
                    XAI_ERR_NORM, "Accumulator shift value = %hhu\nThe accumulator shift value should be less than 32",
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                                          \
                    XAI_ERR_NORM, "Output shift = %hhu\nThe output shift value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#if DILATED_VQ_CONV_S16 == VQ_TRUE
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM4(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM4(coeffTile));
    XAI_CHECK_ERROR((((uintptr_t) (XAI_ARRAY_GET_DATA_PTR(outputScaleArray)) & \
                      0x1) == 0), XAI_ERR_NORM, "The output scale array is not aligned to 2 byte boundary");
#endif
  }

#if DILATED_VQ_CONV_S16 == VQ_FALSE
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t inW = XAI_TILE3D_GET_DIM1(inTile) + \
                      XAI_TILE3D_GET_DIM1_EDGE1(inTile) + XAI_TILE3D_GET_DIM1_EDGE2(inTile);
  const int32_t outW     = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM3(outTile);

  /* Kernel Size (WHDN)*/
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM1(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM2(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Pitches of Coefficient Data (WHDN) */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (WHD) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (WHD) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Data Pointers of input, output, coefficient and bias data */
  int16_t* pInData     = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t* pOutData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t* pBiasData64 = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
  int16_t* pCoeffData  = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
#if DILATED_VQ_CONV_S16 == VQ_TRUE
  uint16_t* restrict pOutScaleData = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  int32_t leftEdge, topEdge;
  int32_t minLim, maxLim;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }


  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MIN : 0;
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX : USHRT_MAX;
  }

  /* Variable Declarations */
  int32_t outCh, x, y, ky, inCh;

  xb_vecNx16 * restrict pvecIn1;
  xb_vecNx16 * restrict pvecIn2;
  xb_vecNx16* restrict pvecOut;
  xb_vecN_2x32v* restrict phvecCoeff1, *restrict phvecCoeff2;
  xb_vec2Nx8 *restrict pdvecBias64;

  xb_vec2Nx8 seq1 = IVP_ADD2NX8(IVP_SEQ2NX8(), 2);
  xb_vec2Nx8 seq2 = IVP_ADD2NX8(IVP_SEQ2NX8(), 34);
  seq2 = IVP_MIN2NX8(seq2, 64);
  xb_vec2Nx8 dvecSel = IVP_SEL2NX8I(seq2, seq1, IVP_SELI_8B_INTERLEAVE_1_LO);
  /* Variable Declarations */
  const int32_t vectorizationWidth = XCHAL_IVPN_SIMD_WIDTH;
  int32_t varLen;
  if (kWidthU > 12)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(32, inW - x);

      for (y = 0; y < outH; y++)    /* Loop across Output height */
      {
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * (y) + (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum21;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12, vecInData11A;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);


              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);


              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 4));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 6));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 6));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 7));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 7));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut3L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);


          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 8)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(32, inW - x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* handles odd output row */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * (y) + (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel*/
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12, vecInData11A;
            xb_vecNx16 vecInData21, vecInData22, vecInData21A;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads 2nd input row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 4));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 4));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 5));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 4)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(32, inW - x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* handles odd output row */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);

        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * (y) + (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12, vecInData11A;
            xb_vecNx16 vecInData21, vecInData22, vecInData21A;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads 2nd input row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              /* multiples loaded input data with first four coeff */

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 3));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(32, inW - x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* handles odd output row */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);

        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * (y) + (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12, vecInData11A;
            xb_vecNx16 vecInData21, vecInData22, vecInData21A;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + inDataPitch1 * enable2ndRow);

#ifdef IS_VISION_130
            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              IVP_L2UNX16_XP(vecInData11, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_L2UNX16_XP(vecInData12, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads 2nd input row */
              IVP_L2UNX16_XP(vecInData21, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_L2UNX16_XP(vecInData22, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));
            } /* for (ky = 0; ky < kHeightU; ky++)*/

#else
            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads 2nd input row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              vecInData11A = IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_8B_ROTATE_RIGHT_2);
              vecInData21A = IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_8B_ROTATE_RIGHT_2);
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData11A, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData21A, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));
            } /* for (ky = 0; ky < kHeightU; ky++)*/
#endif
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  return(XAI_ERROR_STATUS());
}

/******************************************************************************
 *   xaiConvolved(VQ)3D_S_MxNj2d1_S16S16I16_MOW_WHD
 *  ****************************************************************************/
/******************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution   */
/*               with stride = 2. Code implementation is generated during     */
/*               preprocessing stage. This method can be used to generate     */
/*               MxN 3D dilated convolution function and MxN 3D VQ dilated    */
/*               convolution function for S16 bit input data with input stride*/
/*               equal to 1                                                   */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               Output scale array, CNN convolution params structure         */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S16                                             */
/*               biasArray is signed 64, value not exceeding signed 48b       */
/*               Output scale array is U16                                    */
/*               OutData is S16 / U16                                         */
/*               Kernel Size is MxNxDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/****************** xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD ******************/
/****************** xaiConvolvedVQ3D_S_MxNj2d1_S16S16I16_MOW_WHD ****************/

XAI_ERR_TYPE MAKE_NAME(MAKE_NAME_VQ(xaiConvolved, 3D_S_MxNj2d1), S16I16_MOW_WHD) MAKE_ARGUMENTS(inTile, coeffTile, biasArray, outTile, param)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_I16(outTile);
    XAI_CHECK_TILE4D_S16(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S64(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTile) <= 16) &&                   \
                    (XAI_TILE4D_GET_DIM2(coeffTile) <= 16),                     \
                    XAI_ERR_KSIZE, "Kernel Width = %u and Kernel Height = %u\n \
                    Kernel Width or Height should be less than or equal to 16", \
                    XAI_TILE4D_GET_DIM1(coeffTile), XAI_TILE4D_GET_DIM2(coeffTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)), \
                    XAI_ERR_BADARG, "Stride along width = %u and Stride along height = %u\n \
                     Stride along width should be equal to stride along height.",         \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_STRIDE(param, 2);
    XAI_CHECK_DILATION(param, 1);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "Dilation along width = %u and Dilation along height = %u\n \
                     Dilation along width should be equal to dilation along height.",       \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_EDGES_MOW_WHD(inTile, coeffTile, param);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_WHDN);
    XAI_CHECK_CONSISTENCY_MOW_WHD(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_COEFFTILE_CONTIGUOUS(coeffTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 32,                                                         \
                    XAI_ERR_NORM, "Accumulator shift value = %u\nThe accumulator shift value should be less than 32", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                                        \
                    XAI_ERR_NORM, "Output shift = %u\nThe output shift value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#if DILATED_VQ_CONV_S16 == VQ_TRUE
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM4(coeffTile), \
                    XAI_ERR_DATASIZE, "Width of Output Scale Array = %u and Number of Kernels = %u\n \
      Width of Output Scale Array should be greater than or equal to number of kernels.",    \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM4(coeffTile));
    XAI_CHECK_ERROR((((uintptr_t) (XAI_ARRAY_GET_DATA_PTR(outputScaleArray)) & \
                      0x1) == 0), XAI_ERR_NORM, "The output scale array is not aligned to 2 byte boundary");
#endif
  }
#if DILATED_VQ_CONV_S16 == VQ_FALSE
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t inW = XAI_TILE3D_GET_DIM1(inTile) + \
                      XAI_TILE3D_GET_DIM1_EDGE1(inTile) + XAI_TILE3D_GET_DIM1_EDGE2(inTile);
  const int32_t outW     = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM3(outTile);

  /* Kernel Size (WHDN)*/
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM1(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM2(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Pitches of Coefficient Data (WHDN) */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (WHD) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (WHD) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Data Pointers of input, output, coefficient and bias data */
  int16_t* pInData     = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t* pOutData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t* pBiasData64 = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
  int16_t* pCoeffData  = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);

#if DILATED_VQ_CONV_S16 == VQ_TRUE
  uint16_t* restrict pOutScaleData = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  int32_t leftEdge, topEdge;
  int32_t minLim, maxLim;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MIN : 0;
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX : USHRT_MAX;
  }

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  int32_t varLen;

  xb_vecNx16 * restrict pvecIn1;
  xb_vecNx16 * restrict pvecIn2;
  xb_vecNx16* restrict pvecOut;
  xb_vecN_2x32v* restrict phvecCoeff1, *restrict phvecCoeff2;
  xb_vec2Nx8 *restrict pdvecBias64;

  xb_vec2Nx8 seq1 = IVP_ADD2NX8(IVP_SEQ2NX8(), 1);
  xb_vec2Nx8 seq2 = IVP_ADD2NX8(IVP_SEQ2NX8(), 33);
  seq2 = IVP_MIN2NX8(seq2, 64);
  xb_vec2Nx8 dvecSel = IVP_SEL2NX8I(seq2, seq1, IVP_SELI_8B_INTERLEAVE_1_LO);

  /* Number of output elements that can be generated
   * with 2 input vector loads(32 way).*/
  const int32_t vectorizationWidth = (((2 * XCHAL_IVPN_SIMD_WIDTH) - kWidthU) / stride) + 1;

  if (kWidthU > 12)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height  */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);

        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(3rd) input row, corresponding to 2nd output row */

              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* Re-arrange the data in the desired format                                    */
              /* Assume input as 1,2,3,4,5,6,7...64                                          */
              /* After re-arrangement using DSEL operation, updated vectors would be */
              /* vecInData1 : 1,  3,  5,...61                                              */
              /* vecInData2 : 2,  4,  6,...62                                              */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, IVP_SEQ2NX8());
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 4));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 4));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 5));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 6));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 6));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 6));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 6));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 7));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 7));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 7));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 7));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 8)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)     /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)      /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)     /* Loop across Output depth */
        {
          /* handles odd output channel*/
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)     /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)     /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(3rd) input row, corresponding to 2nd output row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* Re-arrange the data in the desired format                                    */
              /* Assume input as 1,2,3,4,5,6,7...64                                          */
              /* After re-arrangement using DSEL operation, updated vectors would be */
              /* vecInData1 : 1,  3,  5,...61                                              */
              /* vecInData2 : 2,  4,  6,...62                                              */

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, IVP_SEQ2NX8());
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 4));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 4));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 5));
            }   /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }     /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        }   /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }     /* end of for (y = 0; y < outH; y += 2)*/
    }       /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 4)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */

            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(3rd) input row, corresponding to 2nd output row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* Re-arrange the data in the desired format                                    */
              /* Assume input as 1,2,3,4,5,6,7...64                                          */
              /* After re-arrangement using DSEL operation, updated vectors would be */
              /* vecInData1 : 1,  3,  5,...61                                              */
              /* vecInData2 : 2,  4,  6,...62                                              */

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, IVP_SEQ2NX8());
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 2));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 3));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 3));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum12, accSum21, accSum22;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);
          accSum12 = accSum11; accSum22 = accSum21;

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(3rd) input row, corresponding to 2nd output row */

              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* Re-arrange the data in the desired format                                    */
              /* Assume input as 1,2,3,4,5,6,7...64                                          */
              /* After re-arrangement using DSEL operation, updated vectors would be */
              /* vecInData1 : 1,  3,  5,...61                                              */
              /* vecInData2 : 2,  4,  6,...62                                              */

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, IVP_SEQ2NX8());
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 0));

              IVP_DSELNX16(vecInData12, vecInData11, vecInData12, vecInData11, dvecSel);
              IVP_DSELNX16(vecInData22, vecInData21, vecInData22, vecInData21, dvecSel);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecInData12, vecInData11, IVP_EXTRN_2X32(hvecCoeffData21, 1));
              IVP_MULPAN16XR16(accSum12, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum22, vecInData22, vecInData21, IVP_EXTRN_2X32(hvecCoeffData21, 1));
            } /* for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1L, vecOut2L, vecOut3L, vecOut4L;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1L, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2L, accSum12, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3L, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4L, accSum22, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut3L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut4L, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  return(XAI_ERROR_STATUS());
}

/******************************************************************************
 *   xaiConvolved(VQ)3D_S_MxNj4d1_S16S16I16_MOW_WHD
 *  ****************************************************************************/
/******************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution   */
/*               with stride = 4. Code implementation is generated during     */
/*               preprocessing stage. This method can be used to generate     */
/*               MxN 3D dilated convolution function and MxN 3D VQ dilated    */
/*               convolution function for S16 bit input data with input stride*/
/*               equal to 4.                                                  */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               Output scale array, CNN convolution params structure         */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S16                                             */
/*               biasArray is signed 64, value not exceeding signed 48b       */
/*               Output scale array is U16                                    */
/*               OutData is S16 / U16                                         */
/*               Kernel Size is MxNxDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/****************** xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD *********************/
/****************** xaiConvolvedVQ3D_S_MxNj4d1_S16S16I16_MOW_WHD *******************/
XAI_ERR_TYPE MAKE_NAME(MAKE_NAME_VQ(xaiConvolved, 3D_S_MxNj4d1), S16I16_MOW_WHD) MAKE_ARGUMENTS(inTile, coeffTile, biasArray, outTile, param)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_I16(outTile);
    XAI_CHECK_TILE4D_S16(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S64(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTile) <= 16) &&                   \
                    (XAI_TILE4D_GET_DIM2(coeffTile) <= 16),                     \
                    XAI_ERR_KSIZE, "Kernel Width = %u and Kernel Height = %u\n \
                    Kernel Width or Height should be less than or equal to 16", \
                    XAI_TILE4D_GET_DIM1(coeffTile), XAI_TILE4D_GET_DIM2(coeffTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)), \
                    XAI_ERR_BADARG, "Stride along width = %u and Stride along height = %u\n \
                     Stride along width should be equal to stride along height.",         \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_STRIDE(param, 4);
    XAI_CHECK_DILATION(param, 1);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "Dilation along width = %u and Dilation along height = %u\n \
                     Dilation along width should be equal to dilation along height.",       \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_EDGES_MOW_WHD(inTile, coeffTile, param);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_WHDN);
    XAI_CHECK_CONSISTENCY_MOW_WHD(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_COEFFTILE_CONTIGUOUS(coeffTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 32,                                                         \
                    XAI_ERR_NORM, "Accumulator shift value = %u\nThe accumulator shift value should be less than 32", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                                        \
                    XAI_ERR_NORM, "Output shift = %u\nThe output shift value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#if DILATED_VQ_CONV_S16 == VQ_TRUE
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM4(coeffTile), \
                    XAI_ERR_DATASIZE, "Width of Output Scale Array = %u and Number of Kernels = %u\n \
      Width of Output Scale Array should be greater than or equal to number of kernels.",    \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM4(coeffTile));
    XAI_CHECK_ERROR((((uintptr_t) (XAI_ARRAY_GET_DATA_PTR(outputScaleArray)) & \
                      0x1) == 0), XAI_ERR_NORM, "The output scale array is not aligned to 2 byte boundary");
#endif
  }
#if DILATED_VQ_CONV_S16 == VQ_FALSE
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t inW = XAI_TILE3D_GET_DIM1(inTile) + \
                      XAI_TILE3D_GET_DIM1_EDGE1(inTile) + XAI_TILE3D_GET_DIM1_EDGE2(inTile);
  const int32_t outW     = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM3(outTile);

  /* Kernel Size (WHDN)*/
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM1(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM2(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Pitches of Coefficient Data (WHDN) */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (WHD) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (WHD) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Data Pointers of input, output, coefficient and bias data */
  int16_t* pInData     = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t* pOutData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t* pBiasData64 = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
  int16_t* pCoeffData  = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);

#if DILATED_VQ_CONV_S16 == VQ_TRUE
  uint16_t* restrict pOutScaleData = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  int32_t leftEdge, topEdge;
  int32_t minLim, maxLim;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }


  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MIN : 0;
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX : USHRT_MAX;
  }

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  int32_t varLen;

  xb_vecNx16 * restrict pvecIn1;
  xb_vecNx16 * restrict pvecIn2;
  xb_vecNx16* restrict pvecOut;
  xb_vecN_2x32v* restrict phvecCoeff1, *restrict phvecCoeff2;
  xb_vec2Nx8 *restrict pdvecBias64;

  /* Number of output elements that can be generated
   * with 2 input vector loads(32 way).*/
  const int32_t vectorizationWidth = (((2 * XCHAL_IVPN_SIMD_WIDTH) - kWidthU) / stride) + 1;

  if (kWidthU > 12)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum21;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6, vecData7, vecData8;
            xb_vecNx16 vecData9, vecData10, vecData11, vecData12;
            xb_vecNx16 vecData13, vecData14, vecData15, vecData16;
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(5th) input row, corresponding to 2nd output row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* 32 elements from 1st row and 32 elements from 2nd row are concatenated here
               * If 1st input row is 0,1,2,3,...63, and the 2nd input row is
               * 64,65,66,67.........127, Data should be arranged  as
               *
               * vecData1 : 0, 4, 8,...56,60,  64,68,72,...120,124
               * vecData2 : 1, 5, 9,...57,61,  65,69,73,...121,125
               * vecData3 : 2, 6,10,...58,62,  66,70,74,...122,126
               * vecData4 : 3, 7,11,...59,63,  67,71,75,...123,127
               *
               * Lower half of the vectors contain data from 1st output row and
               * upper half of the vectors contain data from 2nd output row.
               */

              IVP_DSELNX16(vecData2, vecData1,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SEQ2NX8());
              IVP_DSELNX16(vecData4, vecData3,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              /* multiples loaded input data with 2nd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData5 = IVP_SELNX16I(0, vecData1, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData6 = IVP_SELNX16I(0, vecData2, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData7 = IVP_SELNX16I(0, vecData3, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData8 = IVP_SELNX16I(0, vecData4, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with 3rd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              /* multiples loaded input data with 4th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData9  = IVP_SELNX16I(0, vecData5, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData10 = IVP_SELNX16I(0, vecData6, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData11 = IVP_SELNX16I(0, vecData7, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData12 = IVP_SELNX16I(0, vecData8, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with 5th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData10, vecData9, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecData10, vecData9, IVP_EXTRN_2X32(hvecCoeffData21, 4));
              /* multiples loaded input data with 6th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData12, vecData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecData12, vecData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData13 = IVP_SELNX16I(0, vecData9, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData14 = IVP_SELNX16I(0, vecData10, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData15 = IVP_SELNX16I(0, vecData11, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData16 = IVP_SELNX16I(0, vecData12, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData14, vecData13, IVP_EXTRN_2X32(hvecCoeffData11, 6));
              IVP_MULPAN16XR16(accSum21, vecData14, vecData13, IVP_EXTRN_2X32(hvecCoeffData21, 6));
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData16, vecData15, IVP_EXTRN_2X32(hvecCoeffData11, 7));
              IVP_MULPAN16XR16(accSum21, vecData16, vecData15, IVP_EXTRN_2X32(hvecCoeffData21, 7));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1Ch, vecOut1L, vecOut1H;
          xb_vecNx16 vecOut2Ch, vecOut2L, vecOut2H;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          vecOut1L = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut2L = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut1H = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          vecOut2H = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut1H, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut2H, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 8)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum21;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6, vecData7, vecData8;
            xb_vecNx16 vecData9, vecData10, vecData11, vecData12;
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(5th) input row, corresponding to 2nd output row */
              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* 32 elements from 1st row and 32 elements from 2nd row are concatenated here
               * If 1st input row is 0,1,2,3,...63, and the 2nd input row is
               * 64,65,66,67.........127, Data should be arranged  as
               *
               * vecData1 : 0, 4, 8,...56,60,  64,68,72,...120,124
               * vecData2 : 1, 5, 9,...57,61,  65,69,73,...121,125
               * vecData3 : 2, 6,10,...58,62,  66,70,74,...122,126
               * vecData4 : 3, 7,11,...59,63,  67,71,75,...123,127
               *
               * Lower half of the vectors contain data from 1st output row and
               * upper half of the vectors contain data from 2nd output row.
               */

              IVP_DSELNX16(vecData2, vecData1,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SEQ2NX8());
              IVP_DSELNX16(vecData4, vecData3,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SEQ2NX8());

              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              /* multiples loaded input data with 2nd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData5 = IVP_SELNX16I(0, vecData1, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData6 = IVP_SELNX16I(0, vecData2, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData7 = IVP_SELNX16I(0, vecData3, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData8 = IVP_SELNX16I(0, vecData4, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with 3rd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              /* multiples loaded input data with 4th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData21, 3));

              /* right rotate the input vectors by 2
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData9  = IVP_SELNX16I(0, vecData5, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData10 = IVP_SELNX16I(0, vecData6, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData11 = IVP_SELNX16I(0, vecData7, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData12 = IVP_SELNX16I(0, vecData8, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with 5th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData10, vecData9, IVP_EXTRN_2X32(hvecCoeffData11, 4));
              IVP_MULPAN16XR16(accSum21, vecData10, vecData9, IVP_EXTRN_2X32(hvecCoeffData21, 4));
              /* multiples loaded input data with 6th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData12, vecData11, IVP_EXTRN_2X32(hvecCoeffData11, 5));
              IVP_MULPAN16XR16(accSum21, vecData12, vecData11, IVP_EXTRN_2X32(hvecCoeffData21, 5));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1Ch, vecOut1L, vecOut1H;
          xb_vecNx16 vecOut2Ch, vecOut2L, vecOut2H;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          vecOut1L = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut2L = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut1H = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          vecOut2H = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut1H, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut2H, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else if (kWidthU > 4)
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum21;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6, vecData7, vecData8;
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(5th) input row, corresponding to 2nd output row */

              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* 32 elements from 1st row and 32 elements from 2nd row are concatenated here
               * If 1st input row is 0,1,2,3,...63, and the 2nd input row is
               * 64,65,66,67.........127, Data should be arranged  as
               *
               * vecData1 : 0, 4, 8,...56,60,  64,68,72,...120,124
               * vecData2 : 1, 5, 9,...57,61,  65,69,73,...121,125
               * vecData3 : 2, 6,10,...58,62,  66,70,74,...122,126
               * vecData4 : 3, 7,11,...59,63,  67,71,75,...123,127
               *
               * Lower half of the vectors contain data from 1st output row and
               * upper half of the vectors contain data from 2nd output row.
               */

              IVP_DSELNX16(vecData2, vecData1,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SEQ2NX8());
              IVP_DSELNX16(vecData4, vecData3,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SEQ2NX8());
              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              /* multiples loaded input data with 2nd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData21, 1));

              /* right rotate the input vectors by 2 elements
               * in order to multiply with next column of
               * coeff in the next iteration
               */
              vecData5 = IVP_SELNX16I(0, vecData1, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData6 = IVP_SELNX16I(0, vecData2, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData7 = IVP_SELNX16I(0, vecData3, IVP_SELI_16B_ROTATE_RIGHT_1);
              vecData8 = IVP_SELNX16I(0, vecData4, IVP_SELI_16B_ROTATE_RIGHT_1);

              /* multiples loaded input data with 3rd two coeff */
              IVP_MULPAN16XR16(accSum11, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData11, 2));
              IVP_MULPAN16XR16(accSum21, vecData6, vecData5, IVP_EXTRN_2X32(hvecCoeffData21, 2));
              /* multiples loaded input data with 4th two coeff */
              IVP_MULPAN16XR16(accSum11, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData11, 3));
              IVP_MULPAN16XR16(accSum21, vecData8, vecData7, IVP_EXTRN_2X32(hvecCoeffData21, 3));
            } /* end of for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1Ch, vecOut1L, vecOut1H;
          xb_vecNx16 vecOut2Ch, vecOut2L, vecOut2H;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          vecOut1L = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut2L = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut1H = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          vecOut2H = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut1H, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut2H, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  else
  {
    /* loop across output channels is unrolled twice
     * to produce two output channels in 1 iteration.
     * Also loop across output height by 2 , thereby
     * producing 4 output vectors simultaneously.
     */
    for (x = 0; x < outW; x += vectorizationWidth)   /* Loop across Output width */
    {
      /* out of bound flag */
      int32_t flag = XT_SALT(XCHAL_IVPN_SIMD_WIDTH, inW - stride * x);

      for (y = 0; y < outH; y += 2)    /* Loop across Output height */
      {
        /* In order to handle odd output height */
        int32_t enable2ndRow = XT_SALT(y, outH - 1);
        /* initialize output data pointer */
        int16_t *pOutput = &pOutData[(y * outDataPitch1 + x)];

        /* initialize input data pointer */
        int16_t *pInput = &pInData[inDataPitch1 * stride * (y) + stride * (x)];

        /* initialize coeff and bias data pointer*/
        int16_t *pCoeff = &pCoeffData[0];
        pdvecBias64 = (xb_vec2Nx8 *) pBiasData64;
        valign vaBias = IVP_LA2NX8_PP(pdvecBias64);

        for (outCh = 0; outCh < numOutCh; outCh += 2)   /* Loop across Output depth */
        {
          /* handles odd output channel */
          int32_t enable2ndCh = XT_SALT(outCh, numOutCh - 1);

          /* wide vectors(accumulators) initialized with bias */
          xb_vecNx48 accSum11, accSum21;
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum11, 1);
          ACC_INIT_BIAS64_MOW_ONEACC(pdvecBias64, vaBias, accSum21, enable2ndCh);

          /* priming of coeff load is done outside the innermost loop*/
          phvecCoeff1 = (xb_vecN_2x32v *) (pCoeff);
          valign vaCoeffData1; vaCoeffData1 = IVP_LAN_2X32_PP(phvecCoeff1);

          phvecCoeff2 = (xb_vecN_2x32v *) (pCoeff + coeffPitch3 * enable2ndCh);
          valign vaCoeffData2; vaCoeffData2 = IVP_LAN_2X32_PP(phvecCoeff2);

          for (inCh = 0; inCh < numInCh; inCh++)   /* Loop across input channels */
          {
            /* variable declarations for input and coeff vectors */
            xb_vecN_2x32v hvecCoeffData11;
            xb_vecN_2x32v hvecCoeffData21;

            /* vecInData11 refers to 1st input row, first 32(or lesser) elements
             * and vecInData12 refers to next few left out elements of the same row
             * required to compute one 32 way output vector(To compute one 32 way
             * output vector, we require 32 + edge1 + edge2 number of input elements)
             */
            xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
            xb_vecNx16 vecInData11, vecInData12;
            xb_vecNx16 vecInData21, vecInData22;

            pvecIn1 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2);
            pvecIn2 = (xb_vecNx16 *) (pInput + inCh * inDataPitch2 + \
                                      stride * inDataPitch1 * enable2ndRow);

            for (ky = 0; ky < kHeightU; ky++)   /* Loop across kernel height */
            {
              /* loads 1st input row */
              valign vaInData = IVP_LANX16_PP(pvecIn1);
              IVP_LANX16_XP(vecInData11, vaInData, pvecIn1, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData12, vaInData, pvecIn1, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* loads Next(5th) input row, corresponding to 2nd output row */

              vaInData = IVP_LANX16_PP(pvecIn2);
              IVP_LANX16_XP(vecInData21, vaInData, pvecIn2, 2 * XCHAL_IVPN_SIMD_WIDTH * flag);
              IVP_LANX16_XP(vecInData22, vaInData, pvecIn2, 2 * (inDataPitch1 - XCHAL_IVPN_SIMD_WIDTH * flag));

              /* 32 elements from 1st row and 32 elements from 2nd row are concatenated here
               * If 1st input row is 0,1,2,3,...63, and the 2nd input row is
               * 64,65,66,67.........127, Data should be arranged  as
               *
               * vecData1 : 0, 4, 8,...56,60,  64,68,72,...120,124
               * vecData2 : 1, 5, 9,...57,61,  65,69,73,...121,125
               * vecData3 : 2, 6,10,...58,62,  66,70,74,...122,126
               * vecData4 : 3, 7,11,...59,63,  67,71,75,...123,127
               *
               * Lower half of the vectors contain data from 1st output row and
               * upper half of the vectors contain data from 2nd output row.
               */

              IVP_DSELNX16(vecData2, vecData1,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_0),
                           IVP_SEQ2NX8());
              IVP_DSELNX16(vecData4, vecData3,
                           IVP_SELNX16I(vecInData22, vecInData21, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SELNX16I(vecInData12, vecInData11, IVP_SELI_16B_EXTRACT_2_OF_4_OFF_2),
                           IVP_SEQ2NX8());
              /* load 1 row of coeff for 1st output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData11, vaCoeffData1, phvecCoeff1, coeffPitch1 * 2);

              /* load 1 row of coeff for 2nd output channel */
              IVP_LAVN_2X32_XP(hvecCoeffData21, vaCoeffData2, phvecCoeff2, coeffPitch1 * 2);

              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData11, 0));
              IVP_MULPAN16XR16(accSum21, vecData2, vecData1, IVP_EXTRN_2X32(hvecCoeffData21, 0));
              /* multiples loaded input data with first two coeff */
              IVP_MULPAN16XR16(accSum11, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData11, 1));
              IVP_MULPAN16XR16(accSum21, vecData4, vecData3, IVP_EXTRN_2X32(hvecCoeffData21, 1));
            } /* for (ky = 0; ky < kHeightU; ky++)*/
          }   /* end of for (inCh = 0; inCh < numInCh; inCh++)*/

          /* Pack, Output Scale, Output Shift and clamping */
          xb_vecNx16 vecOut1Ch, vecOut1L, vecOut1H;
          xb_vecNx16 vecOut2Ch, vecOut2L, vecOut2H;
#if DILATED_VQ_CONV_S16 == VQ_TRUE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            pOutScaleData[outCh], outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            pOutScaleData[outCh + enable2ndCh], outShiftU, minLim, maxLim);
#elif DILATED_VQ_CONV_S16 == VQ_FALSE
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1Ch, accSum11, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2Ch, accSum21, packShiftAccU, \
                                            outScale, outShiftU, minLim, maxLim);
#endif
          /* variable store count */
          varLen = XT_MIN(outW - x, vectorizationWidth);

          vecOut1L = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut2L = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_LO_HALVES);
          vecOut1H = IVP_SELNX16I(0, vecOut1Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          vecOut2H = IVP_SELNX16I(0, vecOut2Ch, IVP_SELI_16B_EXTRACT_HI_HALVES);
          /* Storing the first row , first depth output */
          pvecOut = (xb_vecNx16 *) (pOutput);
          valign vaOutData = IVP_ZALIGN();
          IVP_SAVNX16_XP(vecOut1L, vaOutData, pvecOut, 2 * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the first row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndCh * outDataPitch2);
          IVP_SAVNX16_XP(vecOut2L, vaOutData, pvecOut, 2 * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 1st depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + enable2ndRow * outDataPitch1);
          IVP_SAVNX16_XP(vecOut1H, vaOutData, pvecOut, 2 * enable2ndRow * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          /* Storing the 2nd row , 2nd depth output */
          pvecOut = (xb_vecNx16 *) (pOutput + (enable2ndCh * outDataPitch2 + \
                                               enable2ndRow * outDataPitch1));
          IVP_SAVNX16_XP(vecOut2H, vaOutData, pvecOut, 2 * \
                         enable2ndRow * enable2ndCh * varLen);
          IVP_SAPOSNX16_FP(vaOutData, pvecOut);

          pOutput += 2 * outDataPitch2;
          pCoeff  += 2 * coeffPitch3;
        } /* end of (outCh = 0; outCh < numOutCh; outCh += 2)*/
      }   /* end of for (y = 0; y < outH; y += 2)*/
    }     /* end of for (x = 0; x < outW; x += vectorizationWidth)*/
  }
  return(XAI_ERROR_STATUS());
}
#endif /*if ((XCHAL_VISION_TYPE >= 6))*/
