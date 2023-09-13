#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-install
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-install
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-install
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-install
fi

