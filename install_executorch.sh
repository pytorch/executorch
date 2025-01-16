#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Before doing anything, cd to the directory containing this script.
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null || /bin/true

# Find the names of the python tools to use.
if [[ -z $PYTHON_EXECUTABLE ]];
then
  if [[ -z $CONDA_DEFAULT_ENV ]] || [[ $CONDA_DEFAULT_ENV == "base" ]] || [[ ! -x "$(command -v python)" ]];
  then
    PYTHON_EXECUTABLE=python3
  else
    PYTHON_EXECUTABLE=python
  fi
fi

$PYTHON_EXECUTABLE ./install_requirements.py "$@"

# Exit with the same status as the python script.
exit $?
