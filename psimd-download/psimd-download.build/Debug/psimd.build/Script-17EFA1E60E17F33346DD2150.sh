#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp/psimd-gitupdate.cmake
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp/psimd-gitupdate.cmake
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp/psimd-gitupdate.cmake
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/chenlai/tmp_executorch/executorch/psimd-source
  /Users/chenlai/miniconda/bin/cmake -P /Users/chenlai/tmp_executorch/executorch/psimd-download/psimd-prefix/tmp/psimd-gitupdate.cmake
fi

