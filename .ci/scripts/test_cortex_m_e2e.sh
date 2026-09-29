#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI wrapper: export a model for the Cortex-M backend and run it on the
# Corstone-300 FVP via examples/arm/run.sh. The real work (export, runner
# build, FVP launch, Test_result: PASS/FAIL check) is done by run.sh and
# the run_fvp.sh it invokes.

set -eu

MODEL=$1
TARGET=${2:-cortex-m55}
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
et_root_dir=$(realpath "${script_dir}/../..")

# Quantization is the default for cortex-m targets; run.sh's
# arg parser only recognizes --no_quantize, so we omit any explicit flag.
export ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA=True
bash "${et_root_dir}/examples/arm/run.sh" \
    --model_name="${MODEL}" \
    --target="${TARGET}" \
    --bundleio
