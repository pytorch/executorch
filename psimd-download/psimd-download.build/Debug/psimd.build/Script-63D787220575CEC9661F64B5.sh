#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-mkdir
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-mkdir
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-mkdir
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-download
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src
  /Users/chenlai/miniconda/bin/cmake -E make_directory /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-mkdir
fi

