# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/envvar_base.sh"

# Force Apple Clang to avoid Homebrew LLVM, which doesn't properly handle
# Apple SDK Objective-C framework headers (e.g. NSIntegerMax in NSObjCRuntime.h).
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# Pin Xcode so a runner image update cannot change the wheel unnoticed. Its SDK
# decides which Core ML code is built in. 16.4 is the macOS 15 image's default. The
# 26.0.1 pins in other workflows build PyTorch from source, which this build does not.
export DEVELOPER_DIR=/Applications/Xcode_16.4.app/Contents/Developer
