#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Links the path of `libtorch.so` or include library installed in python
# library, to the output of buck, if pytorch is properly installed.
# This util can be used by buck2 build.

set -e

LIB=$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

SUBPATH=$1
OUT=$2
if [[ -f "$LIB/$SUBPATH" ]] || [[ -d "$LIB/$SUBPATH" ]];
then
    ln -s "$LIB/$SUBPATH" "$OUT"
else
    echo "Error: $LIB/$SUBPATH doesn't exist"
    exit 1
fi
