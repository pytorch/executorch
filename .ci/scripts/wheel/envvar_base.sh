# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

# Ensure that CMAKE_ARGS is defined before referencing it. Defaults to empty
# if not defined.
export CMAKE_ARGS="${CMAKE_ARGS:-}"

# setup.py only adds release dependency metadata for artifacts built by the
# binary wheel jobs. A source install on a release branch must preserve the
# PyTorch build already installed by that checkout's CI job.
export EXECUTORCH_BUILDING_WHEEL=1
export EXECUTORCH_WHEEL_VARIANT=cpu
