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
#if XCHAL_HAVE_VISION              //build only on VISION dsps
/******************************** eltwiseMul_BroadCastDims1_j1 ********************************/
/* Description : Optimized implementation of Broadcast Elementwise Multiplication             */
/*               functionality across first dimension.                                        */
/* Inputs      : inTile1, inTile2, param, pitch values                                        */
/* Outputs     : XI Error Code                                                                */
/* InOuts      : Both InTiles and outTile is signed 32bit                                     */
/* Assumptions : While performing element wise multiplication of two input tiles, edge        */
/*               data is ignored                                                              */
/**********************************************************************************************/
static _XAI_INLINE_ void eltwiseMulS32_BroadCastDims1_AV(const xai_pTile3D inTile1,
                                                               const xai_pTile3D inTile2,
                                                               xai_pTile3D outTile,
                                                               int32_t inTile1Pitch0,
                                                               int32_t inTile2Pitch0,
                                                               int32_t inTile1Pitch1,
                                                               int32_t inTile2Pitch1,
                                                               int32_t inTile1Pitch2,
                                                               int32_t inTile2Pitch2)
{
  /* Get Tile Parameters */
  const int32_t dim1SizeOut   = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t dim2SizeOut   = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t dim3SizeOut   = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  int32_t *pInput1 = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile1);
  int32_t *pInput2 = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile2);
  int32_t *pOutput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  int32_t *restrict outPtr1;
  int32_t *restrict inp1Ptr;
  int32_t *restrict inp2Ptr;

  int32_t *restrict outPtr_z;
  int32_t *restrict inp1Ptr_z;
  int32_t *restrict inp2Ptr_z;

  // Outer Most Loop Pitch Variables
  int32_t oOutPitch = outTilePitch2;
  int32_t oIn1Pitch = inTile1Pitch2;
  int32_t oIn2Pitch = inTile2Pitch2;

  // Middle Loop Pitch Variables
  int32_t mOutPitch = outTilePitch1;
  int32_t mIn1Pitch = inTile1Pitch1;
  int32_t mIn2Pitch = inTile2Pitch1;

  int32_t innerMostLoopCnt = dim1SizeOut;
  int32_t middleLoopCnt    = dim2SizeOut;
  int32_t outerMostLoopCnt = dim3SizeOut;

  if (((inTile2Pitch1 == 0) && (inTile2Pitch2 == 0) &&                                      \
       (dim2SizeOut * inTile1Pitch1 == inTile1Pitch2) && (dim1SizeOut == inTile1Pitch1) &&  \
       (dim2SizeOut * outTilePitch1 == outTilePitch2) && (dim1SizeOut == outTilePitch1)) || \
      ((inTile1Pitch1 == 0) && (inTile1Pitch2 == 0) &&                                      \
       (dim2SizeOut * inTile2Pitch1 == inTile2Pitch2) && (dim1SizeOut == inTile2Pitch1) &&  \
       (dim2SizeOut * outTilePitch1 == outTilePitch2) && (dim1SizeOut == outTilePitch1)))
  {
    innerMostLoopCnt = dim1SizeOut * dim2SizeOut * dim3SizeOut;
    middleLoopCnt    = 1;
    outerMostLoopCnt = 1;

    /* Middle Loop Pitch Variables */
    mIn1Pitch = 0;
    mIn2Pitch = 0;
    mOutPitch = 0;

    /* Outer Most Loop Pitch Variables */
    oOutPitch = 0;
    oIn1Pitch = 0;
    oIn2Pitch = 0;
  }
  else if ((inTile2Pitch1 == 0 && dim1SizeOut == inTile1Pitch1 && dim1SizeOut == outTilePitch1) || \
           (inTile1Pitch1 == 0 && dim1SizeOut == inTile2Pitch1 && dim1SizeOut == outTilePitch1))
  {
    innerMostLoopCnt = dim1SizeOut * dim2SizeOut;
    middleLoopCnt    = dim3SizeOut;
    outerMostLoopCnt = 1;

    /* Middle Loop Pitch Variables */
    mOutPitch = outTilePitch2;
    mIn1Pitch = inTile1Pitch2;
    mIn2Pitch = inTile2Pitch2;

    /* Outer Most Loop Pitch Variables */
    oOutPitch = 0;
    oIn1Pitch = 0;
    oIn2Pitch = 0;
  }

#if defined(IVP_MULN_2X32) || (XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5 || (XCHAL_HAVE_BBENEP == 1)) /*Auto vectorization is done only if S32 mul ISA is available*/
/*Adding KQ8 conditionalization also, as KQ8 doesn't have S32 mul support direct or indirect. Therefore, for KQ8 Auto vec attempt shall fail and plain scalar C code shall be used.*/
  /* Tile1 Dimension 1 broadcasting */
  if (inTile1Pitch0 == 0)
  {
      	  // This loop process dim1, dim2, dim3 in the same order from innermost
    for (z = 0; z < outerMostLoopCnt; z++)
    {
      outPtr_z  = (int32_t *) (pOutput + z * oOutPitch);
      inp1Ptr_z = (int32_t *) (pInput1 + z * oIn1Pitch);
      inp2Ptr_z = (int32_t *) (pInput2 + z * oIn2Pitch);

      for (y = 0; y < middleLoopCnt; y++)
      {
        outPtr1 = (int32_t *) (outPtr_z + y * mOutPitch);
        inp1Ptr = (int32_t *) (inp1Ptr_z + y * mIn1Pitch);
        inp2Ptr = (int32_t *) (inp2Ptr_z + y * mIn2Pitch);

        /* Load Input 1 */
        int32_t InData1 = inp1Ptr[0];

        for (x = 0; x < innerMostLoopCnt; x++)
        {
          int32_t InData2   = inp2Ptr[x];
          outPtr1[x] = InData1 * InData2;
        }
      }
    }
  }
  /* Tile2 Dimension 1 broadcasting */
  else if (inTile2Pitch0 == 0)
  {
    // This loop process dim1, dim2, dim3 in the same order from innermost
    for (z = 0; z < outerMostLoopCnt; z++)
    {
      outPtr_z  = (int32_t *) (pOutput + z * oOutPitch);
      inp1Ptr_z = (int32_t *) (pInput1 + z * oIn1Pitch);
      inp2Ptr_z = (int32_t *) (pInput2 + z * oIn2Pitch);

      for (y = 0; y < middleLoopCnt; y++)
      {
        outPtr1 = (int32_t *) (outPtr_z + y * mOutPitch);
        inp1Ptr = (int32_t *) (inp1Ptr_z + y * mIn1Pitch);
        inp2Ptr = (int32_t *) (inp2Ptr_z + y * mIn2Pitch);

        /* Load Input 2 */
        int32_t InData2 = inp2Ptr[0];

        for (x = 0; x < innerMostLoopCnt; x++)
        {
          int32_t InData1   = inp1Ptr[x];
          outPtr1[x] = InData1 * InData2;
        }
      }
    }
  }
#else
  xb_vecN_2x32v  * restrict pvecIn1;
  xb_vecN_2x32v  * restrict pvecIn2;
  xb_vecN_2x32v * restrict pvecOut;

  xb_vecN_2x32v vecInData1;  /* 1st input tile */
  xb_vecN_2x32v vecInData2;  /* 2nd input tile*/

  const int32_t vectorizationWidth = XCHAL_IVPN_SIMD_WIDTH >> 1;
  valign vaOutData = IVP_ZALIGN();

  /* Tile1 Dimension 1 broadcasting */
  if (inTile1Pitch0 == 0)
  {
    // This loop process dim1, dim2, dim3 in the same order from innermost
    for (z = 0; z < outerMostLoopCnt; z++)
    {
      outPtr_z  = (int32_t *) (pOutput + z * oOutPitch);
      inp1Ptr_z = (int32_t *) (pInput1 + z * oIn1Pitch);
      inp2Ptr_z = (int32_t *) (pInput2 + z * oIn2Pitch);

      for (y = 0; y < middleLoopCnt; y++)
      {
        outPtr1 = (int32_t *) (outPtr_z + y * mOutPitch);
        inp1Ptr = (int32_t *) (inp1Ptr_z + y * mIn1Pitch);
        inp2Ptr = (int32_t *) (inp2Ptr_z + y * mIn2Pitch);

        /* Vector and pointer of Input 2 and output to load and store values */
        pvecIn2 = (xb_vecN_2x32v *) (inp2Ptr);
        valign vaInData2 = IVP_LAN_2X32_PP(pvecIn2);

        pvecOut = (xb_vecN_2x32v *) (outPtr1);

        /* Load Input 1 */
        vecInData1 = (xb_vecN_2x32v) (inp1Ptr[0]);
        for (x = 0; x < innerMostLoopCnt; x += vectorizationWidth)
        {
          /* Vector and pointer of Input 2 and output to load and store values */
          IVP_LAVN_2X32_XP(vecInData2, vaInData2, pvecIn2, (innerMostLoopCnt - x) * 4);

          /* populate wide vectors with product of inputs */
          xb_vecN_2x64w wvecAcc;
          wvecAcc = IVP_MULUSN_2X16X32_0(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(vecInData2)), vecInData1);
          IVP_MULAHN_2X16X32_1(wvecAcc, IVP_MOVNX16_FROMN_2X32(vecInData2), vecInData1);

          /* truncate the multiply result in wide vector into 32 bit format*/
          xb_vecN_2x32v vecOutData = IVP_PACKLN_2X64W(wvecAcc);
  
          IVP_SAVN_2X32_XP(vecOutData, vaOutData, pvecOut, (innerMostLoopCnt - x) * 4);
        }
        IVP_SAPOSN_2X32_FP(vaOutData, pvecOut);
      }
    }
  }
  else if (inTile2Pitch0 == 0)
  {
    // This loop process dim1, dim2, dim3 in the same order from innermost
    for (z = 0; z < outerMostLoopCnt; z++)
    {
      outPtr_z  = (int32_t *) (pOutput + z * oOutPitch);
      inp1Ptr_z = (int32_t *) (pInput1 + z * oIn1Pitch);
      inp2Ptr_z = (int32_t *) (pInput2 + z * oIn2Pitch);

      for (y = 0; y < middleLoopCnt; y++)
      {
        outPtr1 = (int32_t *) (outPtr_z + y * mOutPitch);
        inp1Ptr = (int32_t *) (inp1Ptr_z + y * mIn1Pitch);
        inp2Ptr = (int32_t *) (inp2Ptr_z + y * mIn2Pitch);

        /* Vector and pointer of Input 1 and output to load and store values */
        pvecIn1 = (xb_vecN_2x32v *) (inp1Ptr);
        valign vaInData1 = IVP_LAN_2X32_PP(pvecIn1);

        pvecOut = (xb_vecN_2x32v *) (outPtr1);

        /* Load Input 2 */
        vecInData2 = (xb_vecN_2x32v) (inp2Ptr[0]);
        for (x = 0; x < innerMostLoopCnt; x += vectorizationWidth)
        {
          /* load input data from 2nd tile, input data pointer is post incremented by varlen by the load instruction */
          IVP_LAVN_2X32_XP(vecInData1, vaInData1, pvecIn1, (innerMostLoopCnt - x) * 4);

          /* populate wide vectors with product of inputs */
          xb_vecN_2x64w wvecAcc;
          wvecAcc = IVP_MULUSN_2X16X32_0(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(vecInData2)), vecInData1);
          IVP_MULAHN_2X16X32_1(wvecAcc, IVP_MOVNX16_FROMN_2X32(vecInData2), vecInData1);

          /* truncate the multiply result in wide vector into 32 bit format*/
          xb_vecN_2x32v vecOutData = IVP_PACKLN_2X64W(wvecAcc);
  
          IVP_SAVN_2X32_XP(vecOutData, vaOutData, pvecOut, (innerMostLoopCnt - x) * 4);
        }
        IVP_SAPOSN_2X32_FP(vaOutData, pvecOut);
      }
    }
  }
#endif
}

/**************************** xaiEltwiseMul3D ********************************************/
/* Description  : auto-vectorizable implementation of element-wise S32 multiplication    */
/* Inputs       : inTile1, inTile2                                                       */
/* Outputs      : XI Error Code                                                          */
/* InOuts       : outTile                                                                */
/*****************************************************************************************/

XAI_ERR_TYPE xaiEltwiseMul3D_S32_AV(const xai_pTile3D inTile1, const xai_pTile3D inTile2, xai_pTile3D outTile)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S32(inTile1);
    XAI_CHECK_TILE3D_S32(inTile2);
    XAI_CHECK_TILE3D_S32(outTile);
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
  const int32_t inTile1dim1Size = XAI_TILE3D_GET_DIM1(inTile1);
  const int32_t inTile2dim1Size = XAI_TILE3D_GET_DIM1(inTile2);
  const int32_t inTile1dim2Size = XAI_TILE3D_GET_DIM2(inTile1);
  const int32_t inTile2dim2Size = XAI_TILE3D_GET_DIM2(inTile2);
  const int32_t inTile1dim3Size = XAI_TILE3D_GET_DIM3(inTile1);
  const int32_t inTile2dim3Size = XAI_TILE3D_GET_DIM3(inTile2);
  const int32_t dim1SizeOut   = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t dim2SizeOut   = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t dim3SizeOut   = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  int32_t *pInput1 = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile1);
  int32_t *pInput2 = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile2);
  int32_t *pOutput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* broadcast flag is set in case of dimension sizes mismatch of inTile1 and inTile2 */
  /* If broadcast flag is set, only the generalized variant is used, even if edges are absent */
  int32_t bcastFlag = 0;
  if (!((inTile1dim1Size == inTile2dim1Size) && (inTile1dim2Size == inTile2dim2Size) && (inTile1dim3Size == inTile2dim3Size)))
  {
    bcastFlag = 1;
  }

  int32_t is_2D = ((outTilePitch1 == dim1SizeOut) && (XAI_TILE3D_GET_DIM1_PITCH(inTile1) == dim1SizeOut) && (XAI_TILE3D_GET_DIM1_PITCH(inTile2) == dim1SizeOut)) ? 1 : 0;
  int32_t is_1D = ((outTilePitch2 == (dim1SizeOut * dim2SizeOut)) && (XAI_TILE3D_GET_DIM2_PITCH(inTile1) == (dim1SizeOut * dim2SizeOut)) && (XAI_TILE3D_GET_DIM2_PITCH(inTile2) == (dim1SizeOut * dim2SizeOut))) ? 1 : 0;

  /* Get Pitch appropriate for elementwise broadcast operations */
  XAI_TILE3D_GET_BCAST123_PITCH(inTile1, inTile2, inTile1Pitch0, inTile2Pitch0, inTile1Pitch1, \
                                inTile2Pitch1, inTile1Pitch2, inTile2Pitch2);

  if ((inTile1dim1Size == 1 || inTile2dim1Size == 1) && (!(inTile1dim1Size == inTile2dim1Size)))
  {
    eltwiseMulS32_BroadCastDims1_AV(inTile1, inTile2, outTile, inTile1Pitch0, inTile2Pitch0, inTile1Pitch1, \
                                          inTile2Pitch1, inTile1Pitch2, inTile2Pitch2);
  }
  else
  {
#if defined(IVP_MULN_2X32) || (XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5 || (XCHAL_HAVE_BBENEP == 1)) /*Auto vectorization is done only if S32 mul ISA is available*/ 
/*Adding KQ8 conditionalization also, as KQ8 doesn't have S32 mul support direct or indirect. Therefore, for KQ8 Auto vec attempt shall fail and plain scalar C code shall be used.*/
  int32_t *__restrict pIn1;
  int32_t *__restrict pIn2;
  int32_t *__restrict pOut;

  /* Overall design approach is split in 2 sections depending on the optimal
   * tile sizes. When the edge length along dimension1 is zero, loops across
   * dimension1 and dimension2 can be merged.
   */

  /* check for optimal tile size i.e edge length along dimension1 is zero */
  if (is_2D && (!bcastFlag))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3SizeOut;
    int32_t maxLoopCount     = dim1SizeOut * dim2SizeOut;

    /* Updated Loop count based on tile dimension configuration */
    if (is_1D)
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/
      dim3MaxLoopCount = 1;       /* Update max loop counter */
      maxLoopCount    *= dim3SizeOut;
    }
    for (int j = 0; j < dim3MaxLoopCount; j++)
    {
      pIn1 = pInput1 + j * inTile1Pitch2;
      pIn2 = pInput2 + j * inTile2Pitch2;
      pOut = pOutput + j * outTilePitch2;
      for(int i = 0; i < maxLoopCount; i++)
      {
        pOut[i] = (int32_t)(pIn1[i] * pIn2[i]);
      }
    }
  }
  else
  {
    int32_t y, z, idx;

    for (z = 0; z < dim3SizeOut; z++)
    {
      int32_t* temp1 = pInput1 + z * inTile1Pitch2;
      int32_t* temp2 = pInput2 + z * inTile2Pitch2;
      int32_t* temp3 = pOutput + z * outTilePitch2;

      for (y = 0; y < dim2SizeOut; y++)
      {
        pIn1 = (temp1 + y * inTile1Pitch1);
        pIn2 = (temp2 + y * inTile2Pitch1);
        pOut = (temp3 + y * outTilePitch1);

        int32_t InData1, InData2;
        for (idx = 0; idx < dim1SizeOut; idx++)
        {
          InData1   = pIn1[idx];
          InData2   = pIn2[idx];
          pOut[idx] = (int32_t) (InData1 * InData2);
        }
      }
    }
  }
#else
  /* Following code is written for P6/P1 as they don't support S32 MUL. As mentioned, it is used when 32b MUL ISA is not available. */
  /* However, P1 has a proto defined for S32 MUL, internally using 32x16 MUL only.             */
  /* Therefore for P1, the above scalar code shall be used which shall be not auto vectorized, */
  /* as compiler cannot find a direct 32b MUL ISA in P1.                                       */
  /* Therefore, P1 shall give a low performance for this API.                                  */

  /* input and output pointers */
  xb_vecN_2x32v * restrict pvecIn1;
  xb_vecN_2x32v * restrict pvecIn2;
  xb_vecN_2x32v * restrict pdvecOut;

  /* loop variables */
  int32_t x, y, z;

  int32_t vectorizationWidth = XCHAL_IVPN_SIMD_WIDTH >> 1;

  valign vaOutData = IVP_ZALIGN();

  /* Overall design approach is split in 2 sections depending on the optimal
   * tile sizes. When the edge length along dimension1 is zero, loops across
   * dimension1 and dimension2 can be merged.
   */

  /* check for optimal tile size i.e edge length along dimension1 is zero */
  if (is_2D && (!bcastFlag))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3SizeOut;
    int32_t maxLoopCount     = dim1SizeOut * dim2SizeOut;

    /* Updated Loop count based on tile dimension configuration */
    if (is_1D)
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/
      dim3MaxLoopCount = 1;       /* Update max loop counter */
      maxLoopCount    *= dim3SizeOut;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      pvecIn1 = (xb_vecN_2x32v *) &pInput1[z * inTile1Pitch2];
      valign vaInData1 = IVP_LAN_2X32_PP (pvecIn1);

      pvecIn2 = (xb_vecN_2x32v *) &pInput2[z * inTile2Pitch2];
      valign vaInData2 = IVP_LAN_2X32_PP (pvecIn2);

      pdvecOut = (xb_vecN_2x32v *) &pOutput[z * outTilePitch2];

      /* loop across dimension1, dimension2 and dimension3 is combined */
      for (x = 0; x <= maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* input data vectors */
        xb_vecN_2x32v vecInData1;  /* first input tile */
        xb_vecN_2x32v vecInData2;  /* 2nd input tile*/

        /* load input data from 1st tile, input data pointer is post incremented
         * implicitly by SIMD/2 by the load instruction */
        IVP_LAN_2X32_IP(vecInData1, vaInData1, pvecIn1);

        /* load input data from 2nd tile, input data pointer is post incremented
         * implicitly by SIMD/2 by the load instruction */
        IVP_LAN_2X32_IP(vecInData2, vaInData2, pvecIn2);

        /* populate wide vectors with product of inputs */
        xb_vecN_2x64w acc1;
        acc1 = IVP_MULUSN_2X16X32_0(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(vecInData2)), vecInData1);
        IVP_MULAHN_2X16X32_1(acc1, IVP_MOVNX16_FROMN_2X32(vecInData2), vecInData1);
        
        /* truncate the multiply result in wide vector into 32 bit format*/
        xb_vecN_2x32v vecOutData = IVP_PACKLN_2X64W(acc1);

        IVP_SAVN_2X32_XP(vecOutData, vaOutData, pdvecOut, vectorizationWidth * 4);
      } /* end of for (x = 0; x <= maxLoopCount - vectorizationWidth; x += vectorizationWidth) */

      if (x < maxLoopCount)
      {
        /* input data vectors */
        xb_vecN_2x32v vecInData1;  /* 1st input tile */
        xb_vecN_2x32v vecInData2;  /* 2nd input tile*/

        /* variable store count for output */
        int32_t varLen = (maxLoopCount - x) * 4;

        /* load input data from 1st tile, input data pointer is post incremented by varLen, by the load instruction */
        IVP_LAVN_2X32_XP(vecInData1, vaInData1, pvecIn1, varLen);

        /* load input data from 2nd tile, input data pointer is post incremented by varLen, by the load instruction */
        IVP_LAVN_2X32_XP(vecInData2, vaInData2, pvecIn2, varLen);

        /* populate wide vectors with product of inputs */
        xb_vecN_2x64w acc1;
        acc1 = IVP_MULUSN_2X16X32_0(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(vecInData2)), vecInData1);
        IVP_MULAHN_2X16X32_1(acc1, IVP_MOVNX16_FROMN_2X32(vecInData2), vecInData1);

        /* truncate the multiply result in wide vector into 32 bit format*/
        xb_vecN_2x32v vecOutData = IVP_PACKLN_2X64W(acc1);
        IVP_SAVN_2X32_XP(vecOutData, vaOutData, pdvecOut, varLen);
      } /*end of if (x < maxLoopCount)*/
      IVP_SAPOSN_2X32_FP(vaOutData, pdvecOut);
    } /* end of for (z = 0; z < dim3MaxLoopCount; z++) */
  }   /* end of if ((inTile1Pitch1 == dim1SizeOut) && (inTile2Pitch1 == dim1SizeOut) && (outTilePitch1 == dim1SizeOut)) */
  /* Handle cases with edges and/or broadcast along dim2/3 */
  else
  {
    for (x = 0; x < dim1SizeOut; x += vectorizationWidth) /* along 1st dimension */
    {
      /* variable store count for output */
      int32_t varLen = (dim1SizeOut - x) * 4;

      for (z = 0; z < dim3SizeOut; z++)   /* along 3rd dimension */
      {
        int32_t * pIn1 = &pInput1[z * inTile1Pitch2 + x];

        int32_t * pIn2 = &pInput2[z * inTile2Pitch2 + x];
        /* pointer for 1st tile */
        pvecIn1 = (xb_vecN_2x32v *) pIn1;

        /* pointer for 2nd tile */
        pvecIn2 = (xb_vecN_2x32v *) pIn2;

        int32_t * pOut = &pOutput[z * outTilePitch2 + x];

        for (y = 0; y < dim2SizeOut; y++) /* along 2nd dimension */
        {
          /* input data vectors */
          /* 1st input tile */
          xb_vecN_2x32v vecInData1;   

          /* 2nd input tile */
          xb_vecN_2x32v vecInData2;

          /* load input data from 1st tile */
          valign vaInData1 = IVP_LAN_2X32_PP(pvecIn1);

          IVP_LAN_2X32_XP(vecInData1, vaInData1, pvecIn1, inTile1Pitch1 * 4);

          /* load input data from 2nd tile */
          valign vaInData2 = IVP_LAN_2X32_PP (pvecIn2);

          IVP_LAN_2X32_XP(vecInData2, vaInData2, pvecIn2, inTile2Pitch1 * 4);

          /* populate wide vectors with product of inputs */
          xb_vecN_2x64w acc1;
          acc1 = IVP_MULUSN_2X16X32_0(IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(vecInData2)), vecInData1);
          IVP_MULAHN_2X16X32_1(acc1, IVP_MOVNX16_FROMN_2X32(vecInData2), vecInData1);

          pdvecOut = (xb_vecN_2x32v *) pOut;
          /* truncate the multiply result in wide vector into 32 bit format*/
          xb_vecN_2x32v vecOutData = IVP_PACKLN_2X64W(acc1);
          IVP_SAVN_2X32_XP(vecOutData, vaOutData, pdvecOut, varLen);

          IVP_SAPOSN_2X32_FP(vaOutData, pdvecOut);
          pOut += outTilePitch1;
        } /* end of for (y = 0; y < dim2SizeOut; y++) loop */
      }   /* end of for (z = 0; z < dim3SizeOut; z++) loop */
    }   /* end of for (x = 0; x < dim1SizeOut; x += vectorizationWidth) loop */
  } /* end of else */
#endif
  }
  return(XAI_ERROR_STATUS());
}
#endif //#if XCHAL_HAVE_VISION
