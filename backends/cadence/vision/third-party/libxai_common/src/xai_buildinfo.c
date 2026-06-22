/*
 * Copyright (c) 2021 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#include "xai_core.h"

// XI library build configuration - Release/Debug or other
char XAI_BUILD_CONFIGURATION[] = XAI_AUX_STR(_XAI_BUILD_CONFIGURATION_);

// XTRENSA tools version
char XAI_BUILD_TOOLS_VERSION[] = XAI_AUX_STR(_XAI_BUILD_TOOLS_VERSION_);

// target core name and hardware name
char XAI_BUILD_CORE_ID[] =
#if defined(XCHAL_CORE_ID) && defined(XCHAL_HW_VERSION_NAME)
  XCHAL_CORE_ID " (" XCHAL_HW_VERSION_NAME ")"
#elif defined(XCHAL_CORE_ID)
  XCHAL_CORE_ID
#else
  "CSTUB (x86)"
#endif
;

// error level
char XAI_BUILD_ERROR_LEVEL[] =
#if XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR
  "PRINT_AND_CONTINUE_ON_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR) ")"
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_ON_ERROR
  "PRINT_ON_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_PRINT_ON_ERROR) ")"
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_CONTINUE_ON_ERROR
  "CONTINUE_ON_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_CONTINUE_ON_ERROR) ")"
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_RETURN_ON_ERROR
  "RETURN_ON_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_RETURN_ON_ERROR) ")"
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_TERMINATE_ON_ERROR
  "TERMINATE_ON_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_TERMINATE_ON_ERROR) ")"
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_NO_ERROR
  "NO_ERROR (" XAI_AUX_STR(XAI_ERROR_LEVEL_NO_ERROR) ")"
#else
  XAI_AUX_STR(XAI_ERROR_LEVEL)
#endif
;

// library features
char XAI_BUILD_FEATURES_STR[] = ""
#if __XTENSA__ && XAI_EMULATE_LOCAL_RAM && XAI_ERROR_LEVEL != XAI_ERROR_LEVEL_NO_ERROR
                                "DRAM_CHECK "
#endif
;
