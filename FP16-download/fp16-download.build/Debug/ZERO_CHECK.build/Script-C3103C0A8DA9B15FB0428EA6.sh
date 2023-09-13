#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  make -f /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  make -f /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  make -f /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  make -f /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeScripts/ReRunCMake.make
fi

