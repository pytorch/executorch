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

# Named argument:
# -o: output directory/file
# -f: a list of files/directories that we want to link to the output directory, separated by comma ",".
# These paths need to be in relative path format to the python library path.
while getopts ":o:f:" opt; do
  case $opt in
    o) OUT="$OPTARG"
    ;;
    f) FILES="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs to be one of -o, -f."
    exit 1
    ;;
  esac
done

LIB=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

# delimiter ,
export IFS=","

for SUBPATH in $FILES; do
  if [[ -f "$LIB/$SUBPATH" ]] || [[ -d "$LIB/$SUBPATH" ]];
  then
      ln -s "$LIB/$SUBPATH" "$OUT"
  else
      # NB: If a path doesn't exist, it's ok to skip it here. This is to handle the case
      # of optional PyTorch dependencies like libgomp. They are part of PyTorch nightly
      # wheel (from CentOS), but are not needed when building PyTorch from source
      # (from system libgomp)
      echo "Warning: $LIB/$SUBPATH doesn't exist, skipping..."
  fi
done
