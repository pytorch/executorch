#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/download-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/verify-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/extract-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-download
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/download-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/verify-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/extract-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-download
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/download-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/verify-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/extract-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-download
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/download-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/verify-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/extract-fp16.cmake
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/FP16-download/fp16-prefix/src/fp16-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/fp16-download
fi

