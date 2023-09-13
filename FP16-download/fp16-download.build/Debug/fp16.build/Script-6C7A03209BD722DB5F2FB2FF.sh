#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-done
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-done
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-done
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-complete
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-done
fi

