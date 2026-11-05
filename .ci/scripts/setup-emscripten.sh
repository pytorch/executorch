#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# need version >= 17
install_node() {
    curl --retry 3 --retry-all-errors -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    source "$HOME/.nvm/nvm.sh"
    nvm install 22
}

install_emscripten() {
    # The WebGPU backend's instance API (WGPUInstanceFeatureName /
    # WGPUInstanceDescriptor::requiredFeatures) needs the emdawnwebgpu port
    # pinned by Emscripten 4.0.13 or newer; 4.0.10 ships the older
    # capabilities-based descriptor and does not compile it.
    local emscripten_version="${EMSCRIPTEN_VERSION:-4.0.19}"
    git clone https://github.com/emscripten-core/emsdk.git
    pushd emsdk || return
    ./emsdk install "${emscripten_version}"
    ./emsdk activate "${emscripten_version}"
    source ./emsdk_env.sh
    popd || return
}

install_node
install_emscripten
