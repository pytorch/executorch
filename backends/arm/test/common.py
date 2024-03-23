# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil

# TODO: fixme! These globs are a temporary workaround. Reasoning:
# Running the jobs in _unittest.yml will not work since that environment doesn't
# have the vela tool, nor the tosa_reference_model tool. Hence, we need a way to
# run what we can in that env temporarily. Long term, vela and tosa_reference_model
# should be installed in the CI env.
TOSA_REF_MODEL_INSTALLED = shutil.which("tosa_reference_model")
VELA_INSTALLED = shutil.which("vela")
