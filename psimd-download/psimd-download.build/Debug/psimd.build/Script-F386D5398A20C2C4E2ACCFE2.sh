#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-configure
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-configure
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-configure
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd
  /Users/chenlai/miniconda/bin/cmake -E echo_append
  /Users/chenlai/miniconda/bin/cmake -E touch /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/src/psimd-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/psimd-configure
fi

