#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

buck run fbcode//executorch/kernels/test:summarize_supported_features > fbcode/executorch/kernels/test/supported_features_summary.md
