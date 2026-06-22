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
#include "xai_cnn.h"
#include "xai_intrin.h"
#include "limits.h"

#if ((XCHAL_VISION_TYPE >= 6))

/******************************************************************************************
* MOD DWH variants
******************************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation for MxN MOD_DWH                */
/*               3D convolution for S16for handling cases where             */
/*               kwidth * numInch is not a multiple of 4                    */
/*               Code implementation is generated during preprocessing stage*/
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported for dilation = 1  */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is S16, CoeffData is S16                            */
/*               biasArray is signed 64b, value not exceeding signed 48b    */
/*               Output scale array is U16                                  */
/*               OutData is U16/S16                                         */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_S16
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(const xai_pTile3D inTile,
                                                                                const xai_pTile4D coeffTile,
                                                                                const xai_pArray biasArray,
                                                                                const xai_pArray outputScaleArray,
                                                                                xai_pTile3D outTile,
                                                                                const xai_cnn_conv_params *param)
#else
static _XAI_INLINE_ void convolved3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(const xai_pTile3D inTile,
                                                                              const xai_pTile4D coeffTile,
                                                                              const xai_pArray biasArray,
                                                                              xai_pTile3D outTile,
                                                                              const xai_cnn_conv_params *param)
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int16_t *pInData     = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int16_t *pCoeffData  = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int64_t *pBiasData64 = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

#ifdef DILATED_VQ_CONV_S16
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
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


  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];


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
  int32_t outCh, x, y, ky, k;
  int32_t numIter = kWidthU * numInCh;

  xb_vec2Nx8 *restrict pdvecBias = (xb_vec2Nx8 *) (pBiasData64);
  xb_vecN_2x32v* restrict phvecIn1;
  xb_vecN_2x32v* restrict phvecIn2;
  xb_vecN_2x32v* restrict phvecIn3;
  xb_vecN_2x32v* restrict phvecIn4;

  xb_vecNx16* restrict pvecCoeff;
  xb_vecNx16* restrict pvecOut;

  valign vaOutData = IVP_ZALIGN(), vaBias = IVP_LA2NX8_PP(pdvecBias);

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  XCHAL_IVPN_SIMD_WIDTH*/
    xb_vecNx48 accBias48;
    int32_t remainingOutCh = numOutCh - outCh;
    ACC_INIT_BIAS64_MOD_ONEACC(pdvecBias, vaBias, remainingOutCh, accBias48);
#ifdef DILATED_VQ_CONV_S16
    xb_vecNx16U vecScaleData;
    /*Load output scale values*/
    valign vaScale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16U_XP(vecScaleData, vaScale, pOutScaleData, 2 * remainingOutCh);
#endif

    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int16_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vecNx48 accSum1, accSum2, accSum3, accSum4;
        accSum4 = accSum3 = accSum2 = accSum1 = accBias48;

        /* Input Data and Coeff Data Pointers */
        int16_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int16_t *pCoeff = pCoeffData + outCh;

        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          phvecIn1 = (xb_vecN_2x32v *) (pData + ky * inDataPitch2);
          phvecIn2 = (xb_vecN_2x32v *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          phvecIn3 = (xb_vecN_2x32v *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          phvecIn4 = (xb_vecN_2x32v *) (pData + ky * inDataPitch2 + (strideX * \
                                                                     inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LAN_2X32_PP(phvecIn1);
          valign vaData2 = IVP_LAN_2X32_PP(phvecIn2);
          valign vaData3 = IVP_LAN_2X32_PP(phvecIn3);
          valign vaData4 = IVP_LAN_2X32_PP(phvecIn4);
          /* Pointer for Coefficient Load */
          pvecCoeff = (xb_vecNx16 *) (pCoeff + ky * coeffPitch3);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vecN_2x32v hvecData1; IVP_LAVN_2X32_XP(hvecData1, vaData1, phvecIn1, 8);
            xb_vecN_2x32v hvecData2; IVP_LAVN_2X32_XP(hvecData2, vaData2, phvecIn2, 8);
            xb_vecN_2x32v hvecData3; IVP_LAVN_2X32_XP(hvecData3, vaData3, phvecIn3, 8);
            xb_vecN_2x32v hvecData4; IVP_LAVN_2X32_XP(hvecData4, vaData4, phvecIn4, 8);

            /* Aligned Vector Loads of coefficients */
            xb_vecNx16 vecCoeff1; IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff2; IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff3; IVP_L2UNX16_XP(vecCoeff3, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff4; IVP_L2UNX16_XP(vecCoeff4, pvecCoeff, 2 * coeffPitch1);

            IVP_MULPAN16XR16(accSum1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData1, 0));
            IVP_MULPAN16XR16(accSum2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData2, 0));
            IVP_MULPAN16XR16(accSum3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData3, 0));
            IVP_MULPAN16XR16(accSum4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData4, 0));

            IVP_MULPAN16XR16(accSum1, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData1, 1));
            IVP_MULPAN16XR16(accSum2, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData2, 1));
            IVP_MULPAN16XR16(accSum3, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData3, 1));
            IVP_MULPAN16XR16(accSum4, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData4, 1));
          }   /* End Input Channels */
          if (k < numIter)
          {
            int32_t remInCh = numIter - k;
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligning variable vector load of pixels */
            xb_vecN_2x32v hvecData1; IVP_LAVN_2X32_XP(hvecData1, vaData1, phvecIn1, 2 * remInCh);
            xb_vecN_2x32v hvecData2; IVP_LAVN_2X32_XP(hvecData2, vaData2, phvecIn2, 2 * remInCh);
            xb_vecN_2x32v hvecData3; IVP_LAVN_2X32_XP(hvecData3, vaData3, phvecIn3, 2 * remInCh);
            xb_vecN_2x32v hvecData4; IVP_LAVN_2X32_XP(hvecData4, vaData4, phvecIn4, 2 * remInCh);

            /* Aligned Vector Loads of coefficients */
            xb_vecNx16 vecCoeff1; IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, 2 * coeffPitch1 * enable2);
            xb_vecNx16 vecCoeff2; IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, 2 * coeffPitch1 * enable3);
            xb_vecNx16 vecCoeff3; IVP_L2UNX16_XP(vecCoeff3, pvecCoeff, 2 * coeffPitch1);

            IVP_MULPAN16XR16(accSum1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData1, 0));
            IVP_MULPAN16XR16(accSum2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData2, 0));
            IVP_MULPAN16XR16(accSum3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData3, 0));
            IVP_MULPAN16XR16(accSum4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData4, 0));

            IVP_MULPAN16XR16(accSum1, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData1, 1));
            IVP_MULPAN16XR16(accSum2, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData2, 1));
            IVP_MULPAN16XR16(accSum3, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData3, 1));
            IVP_MULPAN16XR16(accSum4, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData4, 1));
          }
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vecNx16 vecOut1, vecOut2, vecOut3, vecOut4;
#ifdef DILATED_VQ_CONV_S16
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut1, accSum1, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut2, accSum2, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut3, accSum3, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut4, accSum4, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1, accSum1, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2, accSum2, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3, accSum3, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4, accSum4, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
#endif
        /* Store the output dvecOut1 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + outCh);
        IVP_SAVNX16_XP(vecOut1, vaOutData, pvecOut, 2 * remainingOutCh);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut2 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch1) * numX);
        IVP_SAVNX16_XP(vecOut2, vaOutData, pvecOut, 2 * remainingOutCh * numX);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut3 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch2) * numY);
        IVP_SAVNX16_XP(vecOut3, vaOutData, pvecOut, 2 * remainingOutCh * numY);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut4 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY));
        IVP_SAVNX16_XP(vecOut4, vaOutData, pvecOut, 2 * remainingOutCh * numX * numY);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is S16, CoeffData is S16                            */
/*               biasArray is signed 64b, value not exceeding signed 48b    */
/*               Output scale array is S16                                  */
/*               OutData is U16/S16                                         */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               No edges along dimension 1 of inTile                       */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_S16
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                      const xai_pTile4D coeffTile,
                                                      const xai_pArray biasArray,
                                                      const xai_pArray outputScaleArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_conv_params *param)
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                    const xai_pTile4D coeffTile,
                                                    const xai_pArray biasArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_conv_params *param)
#endif
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
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,       \
                    "\nKernel Width = %d, Kernel Height = %d\nKernel Width and Height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM3(coeffTile), XAI_TILE4D_GET_DIM4(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1) ||                                                          \
                    ((XAI_CNN_CONV_GET_DILATION(param) >= 1) &&                                                         \
                     (XAI_CNN_CONV_GET_STRIDEX(param) == 1) && (XAI_CNN_CONV_GET_STRIDEY(param) == 1)), XAI_ERR_BADARG, \
                    "\nDilation = %hhu\nDilation should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and Dilation along height = %hhu\n \
                     Dilation along width should be equal to dilation along height.",
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 32,                                       \
                    XAI_ERR_NORM, "\nAccumulator shift value = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                           \
                    XAI_ERR_NORM, "\nOutput shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);

#ifdef DILATED_VQ_CONV_S16
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV_S16
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && XAI_CNN_CONV_GET_DILATION(param) == 1)
  {
#ifdef DILATED_VQ_CONV_S16
    convolvedVQ3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(inTile, \
                                                         coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU   = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU  = XAI_TILE4D_GET_DIM4(coeffTile);
  const int32_t dilationU = XAI_CNN_CONV_GET_DILATION(param);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int16_t *pInData     = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int16_t *pCoeffData  = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int64_t *pBiasData64 = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV_S16
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKWidthU  = dilationU * (kWidthU - 1) + 1;
  int32_t dilatedKHeightU = dilationU * (kHeightU - 1) + 1;
  int32_t leftEdge, topEdge;
  int32_t minLim, maxLim;

  if ((dilatedKWidthU % 2) != 0)
  {
    leftEdge = dilatedKWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKWidthU / 2) : ((dilatedKWidthU / 2) - 1);
  }

  if ((dilatedKHeightU % 2) != 0)
  {
    topEdge = dilatedKHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKHeightU / 2) : ((dilatedKHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

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
  int32_t outCh, x, y, k, inCh;
  valign vaOutData = IVP_ZALIGN();

  /* Vector data pointers */
  xb_vec2Nx8 *restrict pdvecBias = (xb_vec2Nx8 *) (pBiasData64);
  xb_vecN_2x32v* restrict phvecIn1;
  xb_vecN_2x32v* restrict phvecIn2;
  xb_vecN_2x32v* restrict phvecIn3;
  xb_vecN_2x32v* restrict phvecIn4;
  xb_vecNx16* restrict pvecCoeff;
  xb_vecNx16* restrict pvecOut;

  valign vaBias = IVP_LA2NX8_PP(pdvecBias);
  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  XCHAL_IVPN_SIMD_WIDTH*/
    xb_vecNx48 accBias48;
    int32_t remainingOutCh = numOutCh - outCh;
    ACC_INIT_BIAS64_MOD_ONEACC(pdvecBias, vaBias, remainingOutCh, accBias48);
#ifdef DILATED_VQ_CONV_S16
    xb_vecNx16U vecScaleData;
    /*Load output scale values*/
    valign vaScale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16U_XP(vecScaleData, vaScale, pOutScaleData, 2 * remainingOutCh);
#endif

    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int16_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vecNx48 accSum1, accSum2, accSum3, accSum4;
        accSum4 = accSum3 = accSum2 = accSum1 = accBias48;

        /* Input Data and Coeff Data Pointers */
        int16_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int16_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationU - kWidthU * inDataPitch1 * dilationU, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationU);

          /* Pointers for Input Data Loads */
          phvecIn1 = (xb_vecN_2x32v *) (pData + inAddrOff);
          phvecIn2 = (xb_vecN_2x32v *) (pData + inAddrOff + strideX * inDataPitch1 * numX);
          phvecIn3 = (xb_vecN_2x32v *) (pData + inAddrOff + strideY * inDataPitch2 * numY);
          phvecIn4 = (xb_vecN_2x32v *) (pData + inAddrOff + (strideX * \
                                                             inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LAN_2X32_PP(phvecIn1);
          valign vaData2 = IVP_LAN_2X32_PP(phvecIn2);
          valign vaData3 = IVP_LAN_2X32_PP(phvecIn3);
          valign vaData4 = IVP_LAN_2X32_PP(phvecIn4);

          /* Pointer for Coefficient Load */
          pvecCoeff = (xb_vecNx16 *) (pCoeff + coeffAddrOff);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vecN_2x32v hvecData1; IVP_LAVN_2X32_XP(hvecData1, vaData1, phvecIn1, 8);
            xb_vecN_2x32v hvecData2; IVP_LAVN_2X32_XP(hvecData2, vaData2, phvecIn2, 8);
            xb_vecN_2x32v hvecData3; IVP_LAVN_2X32_XP(hvecData3, vaData3, phvecIn3, 8);
            xb_vecN_2x32v hvecData4; IVP_LAVN_2X32_XP(hvecData4, vaData4, phvecIn4, 8);

            /* Aligned Vector Loads of coefficients */
            xb_vecNx16 vecCoeff1; IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff2; IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff3; IVP_L2UNX16_XP(vecCoeff3, pvecCoeff, 2 * coeffPitch1);
            xb_vecNx16 vecCoeff4; IVP_L2UNX16_XP(vecCoeff4, pvecCoeff, 2 * coeffPitch1);

            IVP_MULPAN16XR16(accSum1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData1, 0));
            IVP_MULPAN16XR16(accSum2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData2, 0));
            IVP_MULPAN16XR16(accSum3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData3, 0));
            IVP_MULPAN16XR16(accSum4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData4, 0));

            IVP_MULPAN16XR16(accSum1, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData1, 1));
            IVP_MULPAN16XR16(accSum2, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData2, 1));
            IVP_MULPAN16XR16(accSum3, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData3, 1));
            IVP_MULPAN16XR16(accSum4, vecCoeff4, vecCoeff3, IVP_EXTRN_2X32(hvecData4, 1));
          }   /* End Input Channels */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligning variable vector load of pixels */
            xb_vecN_2x32v hvecData1; IVP_LAVN_2X32_XP(hvecData1, vaData1, phvecIn1, 2 * remInCh);
            xb_vecN_2x32v hvecData2; IVP_LAVN_2X32_XP(hvecData2, vaData2, phvecIn2, 2 * remInCh);
            xb_vecN_2x32v hvecData3; IVP_LAVN_2X32_XP(hvecData3, vaData3, phvecIn3, 2 * remInCh);
            xb_vecN_2x32v hvecData4; IVP_LAVN_2X32_XP(hvecData4, vaData4, phvecIn4, 2 * remInCh);

            /* Aligned Vector Loads of coefficients */
            xb_vecNx16 vecCoeff1; IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, 2 * coeffPitch1 * enable2);
            xb_vecNx16 vecCoeff2; IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, 2 * coeffPitch1 * enable3);
            xb_vecNx16 vecCoeff3; IVP_L2UNX16_XP(vecCoeff3, pvecCoeff, 2 * coeffPitch1);

            IVP_MULPAN16XR16(accSum1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData1, 0));
            IVP_MULPAN16XR16(accSum2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData2, 0));
            IVP_MULPAN16XR16(accSum3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData3, 0));
            IVP_MULPAN16XR16(accSum4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecData4, 0));

            IVP_MULPAN16XR16(accSum1, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData1, 1));
            IVP_MULPAN16XR16(accSum2, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData2, 1));
            IVP_MULPAN16XR16(accSum3, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData3, 1));
            IVP_MULPAN16XR16(accSum4, 0, vecCoeff3, IVP_EXTRN_2X32(hvecData4, 1));
          }
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vecNx16 vecOut1, vecOut2, vecOut3, vecOut4;
#ifdef DILATED_VQ_CONV_S16
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut1, accSum1, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut2, accSum2, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut3, accSum3, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut4, accSum4, packShiftAccU, \
                                             vecScaleData, outShiftU, minLim, maxLim);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1, accSum1, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2, accSum2, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3, accSum3, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4, accSum4, packShiftAccU, \
                                          outScale, outShiftU, minLim, maxLim);
#endif
        /* Store the output dvecOut1 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + outCh);
        IVP_SAVNX16_XP(vecOut1, vaOutData, pvecOut, 2 * remainingOutCh);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut2 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch1) * numX);
        IVP_SAVNX16_XP(vecOut2, vaOutData, pvecOut, 2 * remainingOutCh * numX);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut3 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch2) * numY);
        IVP_SAVNX16_XP(vecOut3, vaOutData, pvecOut, 2 * remainingOutCh * numY);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);

        /* Store the output dvecOut4 along the output depth */
        pvecOut = (xb_vecNx16 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY));
        IVP_SAVNX16_XP(vecOut4, vaOutData, pvecOut, 2 * remainingOutCh * numX * numY);
        IVP_SAPOSNX16_FP(vaOutData, pvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/******************************* end of VQ MOD variants ***************************************/
/**********************************************************************************************/
#endif /*#if ((XCHAL_VISION_TYPE >= 6))*/

