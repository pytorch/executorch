#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake --preset llm \
              -DEXECUTORCH_BUILD_METAL=ON \
              -DCMAKE_INSTALL_PREFIX=cmake-out \
              -DCMAKE_BUILD_TYPE=Release \
              -DEXECUTORCH_ENABLE_LOGGING=ON \
              -DEXECUTORCH_LOG_LEVEL=Info \
              -Bcmake-out -S.

cmake --build cmake-out -j16 --target install --config Release

cmake -DEXECUTORCH_BUILD_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -Sexamples/models/voxtral \
      -Bcmake-out/examples/models/voxtral/

cmake --build cmake-out/examples/models/voxtral --target voxtral_runner --config Release
