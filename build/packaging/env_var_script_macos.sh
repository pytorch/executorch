# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

# Enable pybindings so that users can execute ExecuTorch programs from python.
export EXECUTORCH_BUILD_PYBIND=1

# Ensure that CMAKE_ARGS is defined before referencing it. Defaults to empty
# if not defined.
export CMAKE_ARGS="${CMAKE_ARGS:-}"

# Link the XNNPACK backend into the pybindings runtime so that users can execute
# ExecuTorch programs that delegate to it.
CMAKE_ARGS="${CMAKE_ARGS} -DEXECUTORCH_BUILD_XNNPACK=ON"

# When building for macOS, link additional backends into the pybindings runtime.
CMAKE_ARGS="${CMAKE_ARGS} -DEXECUTORCH_BUILD_COREML=ON"
CMAKE_ARGS="${CMAKE_ARGS} -DEXECUTORCH_BUILD_MPS=ON"
