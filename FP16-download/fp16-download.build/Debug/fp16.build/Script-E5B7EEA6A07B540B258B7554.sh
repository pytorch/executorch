#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-mkdir
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-mkdir
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-mkdir
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/FP16-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-mkdir
fi

