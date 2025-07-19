#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# need version >= 17
install_node() {
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    source "$HOME/.nvm/nvm.sh"
    nvm install 22
}

install_emscripten() {
    git clone https://github.com/emscripten-core/emsdk.git
    pushd emsdk || return
    ./emsdk install 4.0.10
    ./emsdk activate 4.0.10
    source ./emsdk_env.sh
    popd || return
}

install_node
install_emscripten
