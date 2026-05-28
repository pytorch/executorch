/*
 * Copyright (c) 2023 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#undef DILATED_VQ_CONV_PARTIAL
#include "cnn_dilated_conv_partial_MOD_S16.h"
#endif /*#if ((XCHAL_VISION_TYPE >= 6))*/
