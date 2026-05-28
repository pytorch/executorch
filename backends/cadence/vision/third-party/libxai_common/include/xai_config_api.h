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

#ifndef __XAI_CONFIG_API_H__
#define __XAI_CONFIG_API_H__

#ifndef XAI_REF_ONLY_COMPILATION
#include <xtensa/config/core-isa.h>
#endif

// Contains IVP to BBE mappings
#if (XCHAL_HAVE_BBENEP == 1)
#include <xtensa/tie/xt_ivpn.h>
#endif

#include "xai_cnn_version.h"

#ifndef __XTENSA__
    #if defined(_MSC_VER)
        #pragma warning (disable : 4005 )
    #endif
    #ifdef __cplusplus
        #if defined(_MSC_VER) && (_MSC_VER >= 1900)
            #define restrict  __restrict
        #else
            #define restrict
        #endif
    #endif
    #ifndef XCHAL_NUM_DATARAM
        #define XCHAL_NUM_DATARAM  2
    #endif
#endif

#if !defined(__XTENSA__) || !(defined(XCHAL_HAVE_VISION) || defined(XCHAL_HAVE_BBENEP)) || !(XCHAL_HAVE_VISION || XCHAL_HAVE_BBENEP)
#   define XV_EMULATE_DMA
#endif

// #define XAI_EMULATE_LOCAL_RAM 0
#ifndef XAI_EMULATE_LOCAL_RAM
#  define XAI_EMULATE_LOCAL_RAM  1
#endif

/* XI Library API qualifiers */

#if XAI_EMULATE_LOCAL_RAM && __XTENSA__
#if XCHAL_NUM_DATARAM == 2
#  define _XAI_LOCAL_RAM0_  __attribute__((section(".dram0.data")))
#  define _XAI_LOCAL_RAM1_  __attribute__((section(".dram1.data")))
#elif XCHAL_NUM_DATARAM == 1
#  define _XAI_LOCAL_RAM0_  __attribute__((section(".dram0.data")))
#endif
#  define _XAI_LOCAL_IRAM_  __attribute__((section(".iram0.text")))
#else
#  define _XAI_LOCAL_RAM0_
#  define _XAI_LOCAL_RAM1_
#  define _XAI_LOCAL_IRAM_
#endif

#if !defined(_XAI_EXPORTS_)
#  if defined __GNUC__ && __GNUC__ >= 4
#    define _XAI_EXPORTS_  __attribute__((visibility("default")))
#  elif defined(_MSC_VER)
#    if defined(XAI_CREATE_SHARED_LIBRARY)
#      define _XAI_EXPORTS_  __declspec(dllexport)
#    else
#      define _XAI_EXPORTS_  __declspec(dllimport)
#    endif
#  else
#    define _XAI_EXPORTS_
#  endif
#endif

#ifdef __cplusplus
#  define _XAI_EXTERN_C_  extern "C"
#else
#  define _XAI_EXTERN_C_  extern
#endif

#ifdef __cplusplus
#  define XAI_DEFAULT(value) = (value)
#else
#  define XAI_DEFAULT(value)
#endif

#if defined(__XTENSA__) && (!defined(DISABLE_AGGRESSIVE_INLINE))
#define _XAI_INLINE_  __attribute((always_inline))
#else
#define _XAI_INLINE_
#endif

#ifdef GLOW_SPECIAL_BUILD
#   define _XAI_API_      _XAI_EXTERN_C_
#   define _XAI_API_VAR_  _XAI_API_
#else
#   define _XAI_API_      _XAI_EXTERN_C_ _XAI_EXPORTS_ _XAI_INLINE_
#   define _XAI_API_VAR_  _XAI_EXTERN_C_ _XAI_EXPORTS_
#endif

/* error check levels */

/* do not check arguments for errors */
#define XAI_ERROR_LEVEL_NO_ERROR                     0
/* call exit(-1) in case of error */
#define XAI_ERROR_LEVEL_TERMINATE_ON_ERROR           1
/* return corresponding error code on error without any processing (recommended)*/
#define XAI_ERROR_LEVEL_RETURN_ON_ERROR              2
/* capture error but attempt continue processing (dangerous!) */
#define XAI_ERROR_LEVEL_CONTINUE_ON_ERROR            3
/* print error message to stdout and return without any processing */
#define XAI_ERROR_LEVEL_PRINT_ON_ERROR               4
/* print error message but attempt continue processing (dangerous!) */
#define XAI_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR  5

#ifndef XAI_ERROR_LEVEL
#  define XAI_ERROR_LEVEL  XAI_ERROR_LEVEL_RETURN_ON_ERROR
#endif
#endif
