#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2025/1
git submodule update --init --recursive
sudo ./install_build_dependencies.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
make -j$(nproc)

cd ..
cmake --install build --prefix dist

source dist/setupvars.sh
cd ../backends/openvino
pip install -r requirements.txt
cd scripts
./openvino_build.sh --enable_python
