#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

buck2 build //xplat/executorch/backends/vulkan:vulkan_gpuinfo --target-platforms=ovr_config//platform/android:arm64-fbsource --show-output -c ndk.static_linking=true -c ndk.debug_info_level=1 -c executorch.event_tracer_enabled=true
