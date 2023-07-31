#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

buck run @fbcode//on_device_ai/Assistant/Jarvis/mode/Harmony_HiFi4_Opus_Tie_5/dev-eh \
    fbcode//on_device_ai/Assistant/Jarvis/min_runtime:min_runtime_size_test -- \
    fbcode/executorch/test/models/linear_out.ff
