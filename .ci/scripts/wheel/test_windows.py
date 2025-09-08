#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import test_base
from examples.models import Backend, Model
from test_base import ModelTest
from typing import List

def map_backend_name(name: str) -> str:
    # Map the backend name to the string used by the Windows test jobs, which use
    # a slightly different convention. This is an artifact of us being mid-update
    # of the model test logic.
    # TODO(gjcomer) Clean this up when we update the model test CI.

    if name == "xnnpack-quantization-delegation":
        return "xnnpack-q8"
    else:
        return name

def run_tests(model_tests: List[ModelTest]) -> None:
    for model_test in model_tests:
        subprocess.run(
            [
                os.path.join(test_base._repository_root_dir(), ".ci/scripts/test_model.ps1"),
                "-ModelName",
                str(model_test.model),
                "-Backend",
                map_backend_name(str(model_test.backend)),
            ],
            check=True,
            cwd=test_base._repository_root_dir(),
        )


if __name__ == "__main__":
    test_base.run_tests(
        model_tests=[
            ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
        ]
    )
