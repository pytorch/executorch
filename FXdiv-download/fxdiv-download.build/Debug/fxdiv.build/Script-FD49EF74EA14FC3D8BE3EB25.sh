#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/tmp/fxdiv-gitclone.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-download
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/tmp/fxdiv-gitclone.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-download
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/tmp/fxdiv-gitclone.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-download
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/tmp/fxdiv-gitclone.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-download
fi

