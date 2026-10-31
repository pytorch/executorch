# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.pass_manager import ExportedProgramPassManager

from .aten_to_esp_pass import AtenToEspPass


class EspPassManager(ExportedProgramPassManager):
    """Run Espressif operator-library transformations on an Edge program."""

    def __init__(self) -> None:
        super().__init__(passes=[AtenToEspPass()])
