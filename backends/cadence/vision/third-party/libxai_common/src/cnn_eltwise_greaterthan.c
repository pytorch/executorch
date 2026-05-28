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


#define ELT_GREATERTHAN_DATA_TYPE  SIGNED8BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#define ELT_GREATERTHAN_DATA_TYPE  UNSIGNED8BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#define ELT_GREATERTHAN_DATA_TYPE  SIGNED16BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#define ELT_GREATERTHAN_DATA_TYPE  UNSIGNED16BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#define ELT_GREATERTHAN_DATA_TYPE  SIGNED32BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#define ELT_GREATERTHAN_DATA_TYPE  UNSIGNED32BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE

#if XCHAL_HAVE_VISION_HP_VFPU == 1
#define ELT_GREATERTHAN_DATA_TYPE  FLOAT16BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE
#endif

#if XCHAL_HAVE_VISION_SP_VFPU == 1
#define ELT_GREATERTHAN_DATA_TYPE  FLOAT32BIT
#include "cnn_eltwise_greaterthan.h"
#undef ELT_GREATERTHAN_DATA_TYPE
#endif


/**************************** xaiEltwiseGreaterThan3D_AV ***************************************/
/* Description  : General API for auto-vectorizable Broadcast element-wise                     */
/*                GREATER operator                                                             */
/*                Calls one of the xaiEltwiseGreaterThan3D_AV functions based on the data type */
/* Inputs       : inTile1, inTile2                                                             */
/* Outputs      : XI Error Code                                                                */
/* InOuts       : outTile                                                                      */
/***********************************************************************************************/

XAI_ERR_TYPE xaiEltwiseGreaterThan3D_AV(const xai_pTile3D inTile1,
                                        const xai_pTile3D inTile2,
                                        xai_pTile3D outTile)
{
  if (!inTile1 || !inTile2 || !outTile)
  {
    return(XAI_ERR_NULLARG);
  }

  if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_S8))
  {
    return(xaiEltwiseGreaterThan3D_S8_AV(inTile1, inTile2, outTile));
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_U8))
  {
    return(xaiEltwiseGreaterThan3D_U8_AV(inTile1, inTile2, outTile));
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_S16))
  {
    return(xaiEltwiseGreaterThan3D_S16_AV(inTile1, inTile2, outTile));
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_U16))
  {
    return(xaiEltwiseGreaterThan3D_U16_AV(inTile1, inTile2, outTile));
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_S32))
  {
    return(xaiEltwiseGreaterThan3D_S32_AV(inTile1, inTile2, outTile));
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_U32))
  {
    return(xaiEltwiseGreaterThan3D_U32_AV(inTile1, inTile2, outTile));
  }

#if XCHAL_HAVE_VISION_HP_VFPU == 1
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_F16))
  {
    return(xaiEltwiseGreaterThan3D_F16_AV(inTile1, inTile2, outTile));
  }
#endif

#if XCHAL_HAVE_VISION_SP_VFPU == 1
  else if (XAI_TILE3D_CHECK_TYPE(inTile1, XAI_F32))
  {
    return(xaiEltwiseGreaterThan3D_F32_AV(inTile1, inTile2, outTile));
  }
#endif

  return(XAI_ERR_OK);
}

