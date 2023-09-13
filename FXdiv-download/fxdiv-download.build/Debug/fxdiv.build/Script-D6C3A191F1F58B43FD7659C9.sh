#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-done
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-done
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-done
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FXdiv-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fxdiv-done
fi

