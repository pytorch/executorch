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

const char* xaiErrStr(XAI_ERR_TYPE code)
{
  switch (code)
  {
    case XAI_ERR_OK:             return("No error");
    case XAI_ERR_IALIGNMENT:     return("Input alignment requirements are not satisfied");
    case XAI_ERR_OALIGNMENT:     return("Output alignment requirements are not satisfied");
    case XAI_ERR_MALIGNMENT:     return("Same modulo alignment requirement is not satisfied");
    case XAI_ERR_BADARG:         return("Function arguments are somehow invalid");
    case XAI_ERR_MEMLOCAL:       return("Tile is not placed in local memory");
    case XAI_ERR_INPLACE:        return("Inplace operation is not supported");
    case XAI_ERR_EDGE:           return("Edge extension size is too small");
    case XAI_ERR_DATASIZE:       return("Input/output tile size is too small or too big or otherwise inconsistent");
    case XAI_ERR_TMPSIZE:        return("Temporary tile size is too small or otherwise inconsistent");
    case XAI_ERR_KSIZE:          return("Filer kernel size is not supported");
    case XAI_ERR_NORM:           return("Invalid normalization divisor or shift value");
    case XAI_ERR_COORD:          return("Tile coordinates are invalid");
    case XAI_ERR_BADTRANSFORM:   return("Transform is singular or otherwise invalid");
    case XAI_ERR_NULLARG:        return("One of required arguments is NULL");
    case XAI_ERR_THRESH_INVALID: return("Threshold value is somehow invalid");
    case XAI_ERR_SCALE:          return("Provided scale factor is not supported");
    case XAI_ERR_OVERFLOW:       return("Tile size can lead to sum overflow");
    case XAI_ERR_NOTIMPLEMENTED: return("The requested functionality is absent in current version of XI Library");
    case XAI_ERR_CHANNEL_INVALID: return("Channel number is somehow invalid");
    case XAI_ERR_DATATYPE:       return("Argument has invalid data type");
    case XAI_ERR_NO_VARIANT:     return("No suitable variant of the function is available");
    case XAI_ERR_CUSTOMACC_PREPARE: return("Preparing custom acc hardware fails");
    case XAI_ERR_CUSTOMACC_EXECUTE: return("Executing ops on custom acc hardware fails");
    case XAI_ERR_CUSTOMACC_REMOVE:  return("Removing a network for custom acc hardware fails");

    case XAI_ERR_POOR_DECOMPOSITION: return("Computed transform decomposition can produce visual artifacts");
    case XAI_ERR_OUTOFTILE:      return("The arguments or results are out of tile");
    case XAI_ERR_OBJECTLOST:     return("Tracked object is lost");
    case XAI_ERR_RANSAC_NOTFOUND: return("Unable to find an appropriate model for RANSAC");
    case XAI_ERR_REPLAY:         return("Repeated function call is required for completion");
  }
  ;
  return("Unknown error");
}

