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

#ifndef __XAI_CNN_VERSION_H__
#define __XAI_CNN_VERSION_H__

#if ((XCHAL_VISION_TYPE >= 6) || (XCHAL_HAVE_BBENEP == 1))
#if (!defined(GLOW_BUILD) && !defined(MLIR_BUILD) && !defined(XNNC_PROJ_MGR_PROJECT))
#include <xtensa/tie/xt_ivpn.h>
#endif
#endif

#if (XCHAL_VISION_TYPE == 6 && XCHAL_VISION_SIMD16 == 8) //VP1, V110

#define XAI_CNN_LIBRARY_DSP_PROCESSOR              P1
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0

#elif (XCHAL_VISION_TYPE == 6) // VP6, V130

#define XAI_CNN_LIBRARY_DSP_PROCESSOR              P6
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0

#elif ((XCHAL_VISION_TYPE == 7) || ((XCHAL_VISION_TYPE == 9) && (XCHAL_IVPN_SIMD_WIDTH == 32)))   //VQ7, V240, V331, NeuroEdge
#define XAI_CNN_LIBRARY_DSP_PROCESSOR              Q7
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0

#elif ((XCHAL_VISION_TYPE >= 8) || ((XCHAL_HAVE_BBENEP == 1) && (XCHAL_BBEN_SIMD_WIDTH == 64))) // VQ8, V240, V341, MathX_240

#define XAI_CNN_LIBRARY_DSP_PROCESSOR              Q8
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0

#elif (XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5) //HiFi

#define XAI_CNN_LIBRARY_DSP_PROCESSOR              HIFI
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0

#else

#define XAI_CNN_LIBRARY_DSP_PROCESSOR              REFF
#define XAI_CNN_LIBRARY_VERSION_MAJOR              2
#define XAI_CNN_LIBRARY_VERSION_MINOR              0
#define XAI_CNN_LIBRARY_VERSION_PATCH              0
#define XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING  0
#endif //if Processor type

#define XAI_AUX_STR_EXP(__A)  #__A
#define XAI_AUX_STR(__A)      XAI_AUX_STR_EXP(__A)
#define XAI_CNN_LIBRARY_VERSION_STR  XAI_AUX_STR(XAI_CNN_LIBRARY_DSP_PROCESSOR) "." XAI_AUX_STR(XAI_CNN_LIBRARY_VERSION_MAJOR) "." XAI_AUX_STR(XAI_CNN_LIBRARY_VERSION_MINOR) "." XAI_AUX_STR(XAI_CNN_LIBRARY_VERSION_PATCH) "." XAI_AUX_STR(XAI_CNN_LIBRARY_VERSION_INTERNAL_TRACKING)
#endif /* __XAI_CNN_VERSION_H__ */
