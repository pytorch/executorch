#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

exec python backends/arm/scripts/public_api_manifest/validate_public_api_manifest.py \
    --all-manifests
