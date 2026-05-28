/*
 * Copyright (c) 2025 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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
#include <string.h>

/* ----------------------------------------------------------------------------------------------------------------------- */
#if XCHAL_HAVE_VISION // Optimized code is called for Vision DSPs
/* ----------------------------------------------------------------------------------------------------------------------- */
#include "cnn_cast_scalar.h"

#ifdef IN_DATA_TYPE
#undef IN_DATA_TYPE
#endif
#ifdef OUT_DATA_TYPE
#undef OUT_DATA_TYPE
#endif

#define IN_DATA_TYPE   UNSIGNED8BIT
#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif // #ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   SIGNED8BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   UNSIGNED16BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   SIGNED16BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   UNSIGNED32BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   SIGNED32BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define IN_DATA_TYPE   UNSIGNED64BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE


#define IN_DATA_TYPE   SIGNED64BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP


#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define IN_DATA_TYPE   FLOAT16BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE
#endif //#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))


#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
#define IN_DATA_TYPE   FLOAT32BIT
#define OUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED8BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  UNSIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED32BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#ifdef IVP_LAVN_4X64U_XP
#define OUT_DATA_TYPE  UNSIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE

#define OUT_DATA_TYPE  SIGNED64BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif //#ifdef IVP_LAVN_4X64U_XP

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
#define OUT_DATA_TYPE  FLOAT16BIT
#include "cnn_cast.h"
#undef OUT_DATA_TYPE
#endif
#undef IN_DATA_TYPE
#endif //#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))

/**************************** xaiCast3D *****************************************/
/* Description  : General API for data casting                                  */
/* Inputs       : inTile                                                        */
/* Outputs      : XAI Error Code                                                */
/* InOuts       : outTile                                                       */
/********************************************************************************/
XAI_ERR_TYPE xaiCast3D(const xai_pTile3D inTile,
                       xai_pTile3D outTile)
{
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(inTile);
    XAI_CHECK_POINTER(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile),
                    XAI_ERR_BADARG, "\nInput Data Order %d and Output Data Order %d are not same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_S8:
        xaiCast3DFromU8ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromU8ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromU8ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromU8ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromU8ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU8ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU8ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromU8ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromU8ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromS8ToU8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromS8ToU16(inTile, outTile);
      case XAI_S16:
        xaiCast3DFromS8ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromS8ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromS8ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS8ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS8ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromS8ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromS8ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U16))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromU16ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromU16ToS8(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromU16ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromU16ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromU16ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU16ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU16ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromU16ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromU16ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromS16ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromS16ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromS16ToU16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromS16ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromS16ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS16ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS16ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromS16ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromS16ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U32))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromU32ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromU32ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromU32ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromU32ToS16(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromU32ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU32ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromU32ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromU32ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromU32ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S32))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromS32ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromS32ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromS32ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromS32ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromS32ToU32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS32ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromS32ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromS32ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromS32ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U64))
  {
#ifdef IVP_LAVN_4X64U_XP
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromU64ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromU64ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromU64ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromU64ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromU64ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromU64ToS32(inTile, outTile);
        break;
      case XAI_S64:
        xaiCast3DFromU64ToS64(inTile, outTile);
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromU64ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromU64ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
#else //#ifdef IVP_LAVN_4X64U_XP
    if (!XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64))
    {
      xaiCast3DScalar_I64(inTile, outTile);
    }
    else
    {
      return(XAI_ERR_DATATYPE);
    }
#endif  //#ifdef IVP_LAVN_4X64U_XP
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S64))
  {
#ifdef IVP_LAVN_4X64U_XP
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromS64ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromS64ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromS64ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromS64ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromS64ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromS64ToS32(inTile, outTile);
        break;
      case XAI_U64:
        xaiCast3DFromS64ToU64(inTile, outTile);
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromS64ToF16(inTile, outTile);
        break;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromS64ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
#else  //#ifdef IVP_LAVN_4X64U_XP
    if (!XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64))
    {
      xaiCast3DScalar_I64(inTile, outTile);
    }
    else
    {
      return(XAI_ERR_DATATYPE);
    }
#endif  //#ifdef IVP_LAVN_4X64U_XP
  }
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F16))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromF16ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromF16ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromF16ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromF16ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromF16ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromF16ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromF16ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromF16ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
      case XAI_F32:
        xaiCast3DFromF16ToF32(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F32))
  {
    switch (XAI_TILE3D_GET_ELEMENT_TYPE(outTile))
    {
      case XAI_U8:
        xaiCast3DFromF32ToU8(inTile, outTile);
        break;
      case XAI_S8:
        xaiCast3DFromF32ToS8(inTile, outTile);
        break;
      case XAI_U16:
        xaiCast3DFromF32ToU16(inTile, outTile);
        break;
      case XAI_S16:
        xaiCast3DFromF32ToS16(inTile, outTile);
        break;
      case XAI_U32:
        xaiCast3DFromF32ToU32(inTile, outTile);
        break;
      case XAI_S32:
        xaiCast3DFromF32ToS32(inTile, outTile);
        break;
      case XAI_U64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromF32ToU64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
      case XAI_S64:
#ifdef IVP_LAVN_4X64U_XP
        xaiCast3DFromF32ToS64(inTile, outTile);
#else
        xaiCast3DScalar_I64(inTile, outTile);
#endif
        break;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1))
      case XAI_F16:
        xaiCast3DFromF32ToF16(inTile, outTile);
        break;
#endif
      default:
        return(XAI_ERR_DATATYPE);
        break;
    }
  }
#endif
  else
  {
    return(XAI_ERR_DATATYPE);
  }
  return(XAI_ERROR_STATUS());
}

/* ----------------------------------------------------------------------------------------------------------------------- */
#else // Call the reference code only for MathX DSPs for now
/* ----------------------------------------------------------------------------------------------------------------------- */
#if ((XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))
static float fp32_from_bits1(uint32_t w)
{
  union
  {
    uint32_t as_bits;
    float    as_value;
  } fp32 = { w };
  return(fp32.as_value);
}

static uint32_t fp32_to_bits1(float f)
{
  union
  {
    float    as_value;
    uint32_t as_bits;
  } fp32 = { f };
  return(fp32.as_bits);
}

static float convert_fp16_to_fp32(uint16_t h)
{
  const uint32_t w                   = (uint32_t) h << 16;
  const uint32_t sign                = w & UINT32_C(0x80000000);
  const uint32_t two_w               = w + w;
  const uint32_t exp_offset          = UINT32_C(0xE0) << 23;
  const float exp_scale              = fp32_from_bits1(UINT32_C(0x7800000));
  const float normalized_value       = fp32_from_bits1((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask          = UINT32_C(126) << 23;
  const float magic_bias             = 0.5f;
  const float denormalized_value     = fp32_from_bits1((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result              = sign |
                                       (two_w < denormalized_cutoff ? fp32_to_bits1(denormalized_value) : fp32_to_bits1(normalized_value));
  return(fp32_from_bits1(result));
}

static uint16_t convert_fp32_to_fp16(float f)
{
  const float scale_to_inf  = fp32_from_bits1(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits1(UINT32_C(0x08800000));
  float base                = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w      = (uint32_t) fp32_to_bits1(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign   = w & UINT32_C(0x80000000);
  uint32_t bias         = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000))
  {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits1((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits          = fp32_to_bits1(base);
  const uint32_t exp_bits      = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign       = exp_bits + mantissa_bits;
  return((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

/**************************** xaiCast3D *****************************************/
/* Description  : General API for data casting                                  */
/* Inputs       : inTile                                                        */
/* Outputs      : XAI Error Code                                                */
/* InOuts       : outTile                                                       */
/********************************************************************************/
XAI_ERR_TYPE xaiCast3D(const xai_pTile3D inTile,
                       xai_pTile3D outTile)
{
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(inTile);
    XAI_CHECK_POINTER(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile),
                    XAI_ERR_BADARG, "\nInput Data Order %d and Output Data Order %d are not same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get tile parameters */
  const int32_t dim1Size  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size  = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inPitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inPitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Input data pointers */
  uint8_t *pIn_8bU   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pIn_8b     = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint16_t *pIn_16bU = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pIn_16b   = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint32_t *pIn_32bU = (uint32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pIn_32b   = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint64_t *pIn_64bU = (uint64_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pIn_64b   = (int64_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
  xb_f16 *pIn_f16b = (xb_f16  *) XAI_TILE3D_GET_DATA_PTR(inTile);
#endif
  float *pIn_f32b = (float  *) XAI_TILE3D_GET_DATA_PTR(inTile);

  /* Output data pointers */
  uint8_t *pOut_8bU   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pOut_8b     = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint16_t *pOut_16bU = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int16_t *pOut_16b   = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint32_t *pOut_32bU = (uint32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t *pOut_32b   = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  uint64_t *pOut_64bU = (uint64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t *pOut_64b   = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
  xb_f16 *pOut_f16b = (xb_f16  *) XAI_TILE3D_GET_DATA_PTR(outTile);
#endif
  float *pOut_f32b = (float  *) XAI_TILE3D_GET_DATA_PTR(outTile);

  uint16_t temp;
  int32_t x, y, z;

  for (z = 0; z < dim3Size; z++) /* along 3rd dimension */
  {
    for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
    {
      for (x = 0; x < dim1Size; x++) /* along 1st dimension */
      {
        // Conversions to U64
        if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> U64
            case XAI_U8:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> U64
            case XAI_S8:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> U64
            case XAI_U16:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> U64
            case XAI_S16:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> U64
            case XAI_U32:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> U64
            case XAI_S32:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U64
            case XAI_S64:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> U64
            case XAI_F32:
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F16 -> U64
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);  // Strict Aliasing Rule, TENX-63685
              pOut_64bU[z * outPitch2 + y * outPitch1 + x] = (uint64_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to S64
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> S64
            case XAI_U8:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> S64
            case XAI_S8:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> S64
            case XAI_U16:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> S64
            case XAI_S16:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> S64
            case XAI_U32:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> S64
            case XAI_S32:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S64
            case XAI_U64:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> S64
            case XAI_F32:
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F16 -> S64
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_64b[z * outPitch2 + y * outPitch1 + x] = (int64_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to U32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> U32
            case XAI_U8:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> U32
            case XAI_S8:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> U32
            case XAI_U16:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> U32
            case XAI_S16:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> U32
            case XAI_S32:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> U32
            case XAI_U64:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U32
            case XAI_S64:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> U32
            case XAI_F32:
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F16 -> U32
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_32bU[z * outPitch2 + y * outPitch1 + x] = (uint32_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to S32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> S32
            case XAI_U8:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> S32
            case XAI_S8:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> S32
            case XAI_U16:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> S32
            case XAI_S16:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> S32
            case XAI_U32:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> S32
            case XAI_U64:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S32
            case XAI_S64:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> S32
            case XAI_F32:
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F16 -> S32
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_32b[z * outPitch2 + y * outPitch1 + x] = (int32_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to U16
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> U16
            case XAI_U8:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> U16
            case XAI_S8:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> U16
            case XAI_S16:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> U16
            case XAI_U32:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> U16
            case XAI_S32:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> U16
            case XAI_U64:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U16
            case XAI_S64:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> U16
            case XAI_F32:
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F32 -> U16
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_16bU[z * outPitch2 + y * outPitch1 + x] = (uint16_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to S16
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> S16
            case XAI_U8:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> S16
            case XAI_S8:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> S16
            case XAI_U16:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> S16
            case XAI_U32:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> S16
            case XAI_S32:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> S16
            case XAI_U64:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S16
            case XAI_S64:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> S16
            case XAI_F32:
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F32 -> S16
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_16b[z * outPitch2 + y * outPitch1 + x] = (int16_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to U8
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // S8 -> U8
            case XAI_S8:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> U8
            case XAI_U16:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> U8
            case XAI_S16:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> U8
            case XAI_U32:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> U8
            case XAI_S32:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> U8
            case XAI_U64:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> U8
            case XAI_S64:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> U8
            case XAI_F32:
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F32 -> U8
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_8bU[z * outPitch2 + y * outPitch1 + x] = (uint8_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to S8
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> S8
            case XAI_U8:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> S8
            case XAI_U16:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> S8
            case XAI_S16:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> S8
            case XAI_U32:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> S8
            case XAI_S32:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> S8
            case XAI_U64:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> S8
            case XAI_S64:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
            // F32 -> S8
            case XAI_F32:
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) pIn_f32b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F32 -> S8
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_8b[z * outPitch2 + y * outPitch1 + x] = (int8_t) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to F32
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_F32))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> F32
            case XAI_U8:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_8bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S8 -> F32
            case XAI_S8:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_8b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U16 -> F32
            case XAI_U16:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_16bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S16 -> F32
            case XAI_S16:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_16b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U32 -> F32
            case XAI_U32:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_32bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S32 -> F32
            case XAI_S32:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_32b[z * inPitch2 + y * inPitch1 + x];
              break;
            // U64 -> F32
            case XAI_U64:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_64bU[z * inPitch2 + y * inPitch1 + x];
              break;
            // S64 -> F32
            case XAI_S64:
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) pIn_64b[z * inPitch2 + y * inPitch1 + x];
              break;
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
            // F16 -> F32
            case XAI_F16:
              memcpy(&temp, &pIn_f16b[z * inPitch2 + y * inPitch1 + x], 2);
              pOut_f32b[z * outPitch2 + y * outPitch1 + x] = (float) convert_fp16_to_fp32(temp);
              break;
#endif
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
        // Conversions to F16
#if (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)
        else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_F16))
        {
          switch (XAI_TILE3D_GET_ELEMENT_TYPE(inTile))
          {
            // U8 -> F16
            case XAI_U8:
              temp = convert_fp32_to_fp16((float) pIn_8bU[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);             // Strict Aliasing Rule, TENX-63685
              break;
            // S8 -> F16
            case XAI_S8:
              temp = convert_fp32_to_fp16((float) pIn_8b[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // U16 -> F16
            case XAI_U16:
              temp = convert_fp32_to_fp16((float) pIn_16bU[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // S16 -> F16
            case XAI_S16:
              temp = convert_fp32_to_fp16((float) pIn_16b[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // U32 -> F16
            case XAI_U32:
              temp = convert_fp32_to_fp16((float) pIn_32bU[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // S32 -> F16
            case XAI_S32:
              temp = convert_fp32_to_fp16((float) pIn_32b[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // U64 -> F16
            case XAI_U64:
              temp = convert_fp32_to_fp16((float) pIn_64bU[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // S64 -> F16
            case XAI_S64:
              temp = convert_fp32_to_fp16((float) pIn_64b[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            // F32 -> F16
            case XAI_F32:
              temp = convert_fp32_to_fp16((float) pIn_f32b[z * inPitch2 + y * inPitch1 + x]);
              memcpy(&pOut_f16b[z * outPitch2 + y * outPitch1 + x], &temp, 2);
              break;
            default:
              return(XAI_ERR_NO_VARIANT);
              break;
          }
        }
#endif
      } /* end (x = 0; x < dim1Size; x++) loop */
    }   /* end (y = 0; y < dim2Size; y++) loop */
  }     /* end (z = 0; z < dim3Size; z++) loop */

  return(XAI_ERROR_STATUS());
}
#endif // #if (((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_BBENEP_HP_VFPU == 1)) || ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1)))
/* ----------------------------------------------------------------------------------------------------------------------- */
#endif // #if XCHAL_HAVE_VISION
/* ----------------------------------------------------------------------------------------------------------------------- */
