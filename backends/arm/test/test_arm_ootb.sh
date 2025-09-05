#!/usr/bin/env bash

# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

run_ootb_tests_ethos_u() {
    echo "$FUNCNAME: Running out-of-the-box tests for Arm Ethos-U"
    jupyter nbconvert \
        --to notebook \
        --execute examples/arm/ethos_u_minimal_example.ipynb
    echo "${FUNCNAME}: PASS"
}

run_ootb_tests_ethos_u
