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
